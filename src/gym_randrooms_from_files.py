import numpy as np
import os
import time
import random
import rospy
import gym
from gym.spaces import Tuple, Box
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Bool as BoolMsg
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Transform
from PIL import Image


X_MIN, X_MAX, Y_MIN, Y_MAX = 0., 5., 0., 4.
SAFE_MAPPER_LATENT_DIM = 64
MAV_NAME = 'firefly'


class RandomRooms(gym.Env):
    """
    RL environment for exploration of a set of rooms. Each room gets one
    episode of a fixed number of timesteps. The order of the rooms is
    uniformly random.
    """

    def __init__(self, config):
        rospy.init_node('reinforcement_learner', anonymous=True)
        rospy.Subscriber('/' + MAV_NAME + '/vi_sensor/left/image_raw', ImageMsg,
                         self.ros_img_callback)
        rospy.Subscriber('/' + MAV_NAME + '/arrived_flag', BoolMsg,
                         self.ros_arrival_callback)
        self.waypoint_publisher = rospy.Publisher('/firefly/command/trajectory',
                                                  MultiDOFJointTrajectory)

        self.rooms_dir = config['rooms_dir']  # directory containing room files
        self.img_scale = config['img_scale']  # pixels per unit length
        self.rooms_left = set(range(len(os.listdir(self.rooms_dir))))

        self.true_map = None
        self.full_view = None
        self.enter_new_room()

        self.x, self.y, self.t = 0., 0., 0.
        self.in_transit = False
        self.model = SafeMapperLSTM()
        self.ep_len = config['ep_length']  # episode length in actual time

        self.action_space = Box(low=-np.inf, high=np.inf,
                                shape=(2,), dtype=np.float32)
        self.observation_space = Tuple((self.action_space,
                                        Box(low=-np.inf, high=np.inf,
                                            shape=(SAFE_MAPPER_LATENT_DIM,),
                                            dtype=np.float32)))

    def agent_observation(self):
        """Agent's state observation: position and safe-mapper latent code."""
        return [self.x, self.y], self.model.latent_state()

    def enter_new_room(self, room_id=-1):
        """
        Loads the true safe-map and image data for a new room.
        :param room_id: The ID of the room to load. If negative, a random
                        unseen room will be chosen.
        :return: none
        """
        if room_id < 0:
            room_id = random.choice(self.rooms_left)
        self.true_map = np.load(self.rooms_dir + str(room_id) + 'truth.npy')
        self.full_view = np.load(self.rooms_dir + str(room_id) + 'view.npy')
        self.rooms_left.remove(room_id)

    def reset(self):
        """Called at the end of each episode(room)."""
        self.enter_new_room()
        self.x, self.y, self.t = 0., 0., 0.
        return self.agent_observation()

    def step(self, action):
        """
        Simulates the agent performing the given action (movement and query).
        Calculates the reward and updates the state according to the environment
        dynamics. The position becomes the new position and the safe mapper
        is retrained with the new safety label. Absolute time is also updated.

        :param action: array-like with (query x, query y)
        :return: state, reward, done (bool), auxiliary info
        """
        assert len(action) == 2
        reward = self._get_reward(action, 0.1)

        ros_publish_waypoint(action)
        self.in_transit = True
        start_time = time.time()
        while self.in_transit:
            time.sleep(0.1)

        #self.model.train()
        #self.t += self._get_time_penalty(action)
        self.t += time.time() - start_time
        self.x, self.y = action[0], action[1]
        done = self.t >= self.ep_len
        return self.agent_observation(), reward, done, {}

    def _get_reward(self, action, time_weight):
        """Reward consists of model improvement minus time taken, weighted."""
        return self._get_model_improvement(action) - \
               time_weight * self._get_time_penalty(action)

    def _get_time_penalty(self, action):
        """Distance-based estimate of the time needed to perform the action."""
        return np.sqrt((action[0] - self.x) ** 2 + (action[1] - self.y) ** 2)

    def _get_model_improvement(self, action, mode='gl'):
        """Heuristic metric for the improvement of the safe-mapper."""
        # Converting query position to pixel indices
        x_idx = int(action[0] * self.img_scale)
        y_idx = int(action[1] * self.img_scale)
        label = self.true_map[x_idx][y_idx]
        if mode == 'gl':
            return self.model.gradlen(label)
        elif mode == 'lcu':
            return self.model.uncertainty(label)
        else:
            return 0

    def ros_img_callback(self, img_msg):
        """Send image to safe-mapper LSTM upon ROS image message."""
        # Convert to grayscale array
        img = np.array(Image.frombuffer('RGB', (img_msg.width, img_msg.height),
                                        img_msg.data, 'raw', 'L', 0, 1))
        # Feed new image to LSTM

    def ros_arrival_callback(self, arrival_msg):
        """Change flag upon ROS arrival message to allow RL step to continue."""
        assert arrival_msg
        self.in_transit = False

    def ros_publish_waypoint(self, action):
        # create trajectory msg
        traj = MultiDOFJointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.header.frame_id = 'frame'
        traj.joint_names.append('base_link')

        # create end point for trajectory
        transforms = Transform()
        transforms.translation.x = action[0]
        transforms.translation.y = action[1]
        transforms.translation.z = z

        quat = tf.transformations.quaternion_from_euler(yaw * np.pi / 180.0, 0, 0, axes='rzyx')
        transforms.rotation.x = quat[0]
        transforms.rotation.y = quat[1]
        transforms.rotation.z = quat[2]
        transforms.rotation.w = quat[3]

        velocities = Twist()
        accel = Twist()
        point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accel], rospy.Time(1))
        traj.points.append(point)
        self.waypoint_publisher.publish(traj)