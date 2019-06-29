import numpy as np
import os
import time
import random
import rospy
import tf
import gym
from gym.spaces import Tuple, Box
from sensor_msgs.msg import Image as ImageMsg
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Twist, Transform, Pose
from PIL import Image


X_MIN, X_MAX, Y_MIN, Y_MAX = 0., 5., 0., 5.
Z = 3.
DRONE_X_LENGTH, DRONE_Y_LENGTH = 0.6, 0.6
TABLE_X_LENGTH, TABLE_Y_LENGTH = 1., 1.
TABLES_PER_ROOM = 5
SAFE_MAPPER_LATENT_DIM = 6
MAV_NAME = 'firefly'


def random_table_centers(n_tables):
    """
    Generate random table center locations within the bounds of the room and not
    overlapping with each other.

    :param n_tables: The number of table centers to generate
    :return: The table centers in a list of [x, y] pairs
    """
    def is_overlap(centers, new_x, new_y):
        overlap = False
        for x, y in centers:
            if abs(x - new_x) < TABLE_X_LENGTH and abs(y - new_y) < TABLE_Y_LENGTH:
                overlap = True
        return overlap

    centers = []
    for _ in range(n_tables):
        new_x = TABLE_X_LENGTH / 2 + np.random.rand() * (X_MAX - TABLE_X_LENGTH)
        new_y = TABLE_Y_LENGTH / 2 + np.random.rand() * (Y_MAX - TABLE_Y_LENGTH)
        while is_overlap(centers, new_x, new_y):
            new_x = TABLE_X_LENGTH / 2 + np.random.rand() * (X_MAX - TABLE_X_LENGTH)
            new_y = TABLE_Y_LENGTH / 2 + np.random.rand() * (Y_MAX - TABLE_Y_LENGTH)
        centers.append([new_x, new_y])
    return centers

class SafeMapperLSTM:
    def latent_state(self): return np.zeros(SAFE_MAPPER_LATENT_DIM)
    def save_input(self, action, label): pass
    def train_one_step(self, label): pass
    def gradlen(self, label): return -1
    def uncertainty(self): return label

class RandomRooms(gym.Env):
    """
    RL environment for exploration of a set of rooms. Each room gets one
    episode of a fixed number of timesteps. The order of the rooms is
    uniformly random.
    """

    def __init__(self, config):
        rospy.init_node('agent', anonymous=True)
        rospy.Subscriber('/' + MAV_NAME + '/vi_sensor/left/image_raw', ImageMsg,
                         self.ros_img_callback)
        rospy.Subscriber('/' + MAV_NAME + '/ground_truth/pose', Pose,
                         self.ros_pose_callback)
        self.ready_for_img = False
        self.waypoint_publisher = rospy.Publisher('/firefly/command/trajectory',
                                                  MultiDOFJointTrajectory)

        self.x, self.y, self.t = 0., 0., 0.
        self.start_time = time.time()
        self.model = SafeMapperLSTM()
        self.time_weight = config['time_weight']  # coefficient of time (s) in reward
        self.ep_len = config['ep_length']  # episode length in actual time
        self.map_scale = config['map_scale']  # pixels per unit length
        self.img_interval = config['img_interval']  # how far apart each image is (at most)
        self.true_map = None
        #self._enter_new_room(True)
        #self.ros_publish_waypoint([0, 0])
        time.sleep(5.)

        self.action_space = Box(low=np.array([X_MIN, Y_MIN]), high=np.array([X_MAX, Y_MAX]),
                                dtype=np.float32)
        self.observation_space = Tuple((self.action_space,
                                        Box(low=-np.inf, high=np.inf,
                                            shape=(SAFE_MAPPER_LATENT_DIM,),
                                            dtype=np.float32),
                                        Box(low=0, high=np.inf,
                                            shape=(1,), dtype=np.float32)))

    def agent_observation(self):
        """Agent's state observation: position, safe-mapper latent code, time."""
        return [self.x, self.y], self.model.latent_state(), [self.t]

    def _enter_new_room(self, is_first=False):
        """
        Generates a new random room and updates true safe map. Each room contains several
        randomly placed tables and we consider tabletop surfaces as the safe areas.
        """
        # Delete old tables
        if not is_first:
            for i in range(TABLES_PER_ROOM):
                os.system("rosservice call /gazebo/delete_model '{model_name: table" + str(i) + "}'")

        # Spawn new tables
        table_centers = random_table_centers(TABLES_PER_ROOM)
        for i in range(TABLES_PER_ROOM):
            os.system("rosrun gazebo_ros spawn_model -database cafe_table -sdf " +
                      "-model table" + str(i) +
                      " -x " + str(table_centers[i][0]) +
                      " -y " + str(table_centers[i][1]))

        # Calculate new true safe map
        self.true_map = np.zeros((int(X_MAX * self.map_scale), int(Y_MAX * self.map_scale)))
        safe_x_extent = (TABLE_X_LENGTH - DRONE_X_LENGTH) / 2
        safe_y_extent = (TABLE_Y_LENGTH - DRONE_Y_LENGTH) / 2
        for x, y in table_centers:
            self.true_map[int((x - safe_x_extent) * self.map_scale):
                          int((x + safe_x_extent) * self.map_scale),
                          int((y - safe_y_extent) * self.map_scale):
                          int((y + safe_y_extent) * self.map_scale)] = 1

    def reset(self):
        """Called at the end of each episode(room) to enter a new room and reset position."""
        self.ros_publish_waypoint([0, 0])
        self._enter_new_room()
        self.wait_for_arrival([0, 0])
        self.x, self.y, self.t = 0., 0., 0.
        self.start_time = time.time()
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
        label = self._get_label(action)
        model_err = self._get_model_improvement(label)
        self.model.train_one_step(label)

        while True:  # Travel to query destination in small steps
            dist_remain = self._get_distance(action)
            if dist_remain < self.img_interval:
                self.ros_publish_waypoint(action)
                self.wait_for_arrival(action)
                break
            else:
                frac_dist_remain = self.img_interval / dist_remain
                next_x = self.x + (action[0] - self.x) * frac_dist_remain
                next_y = self.y + (action[1] - self.y) * frac_dist_remain
                self.ros_publish_waypoint([next_x, next_y])
                self.wait_for_arrival([next_x, next_y])

        self.t = time.time() - self.start_time
        reward = model_err + self.x * self.y  # - self.time_weight * travel_time
        done = self.t >= self.ep_len
        return self.agent_observation(), reward, done, {}

    def _get_label(self, action):
        """Get the true safe map label at the given position."""
        # Converting query position to pixel indices
        x_idx = int(action[0] * self.map_scale)
        y_idx = int(action[1] * self.map_scale)
        return self.true_map[x_idx][y_idx]

    def _get_reward(self, action):
        """Reward consists of model improvement minus time taken, weighted."""
        return self._get_model_improvement(action) - \
               self.time_weight * self._get_distance(action)

    def _get_distance(self, action):
        """Distance of the action destination from the current position."""
        return np.sqrt((action[0] - self.x) ** 2 + (action[1] - self.y) ** 2)

    def _get_model_improvement(self, label, mode='gl'):
        """Heuristic metric for the improvement of the safe-mapper."""
        if mode == 'gl':
            return self.model.gradlen(label)
        elif mode == 'lcu':
            return self.model.uncertainty(label)
        else:
            return 0

    def ros_img_callback(self, img_msg):
        """Send image to safe-mapper LSTM upon ROS image message."""
        if not self.ready_for_img:
            return
        self.ready_for_img = False

        # Convert to grayscale array
        img = np.array(Image.frombuffer('RGB', (img_msg.width, img_msg.height),
                                        img_msg.data, 'raw', 'L', 0, 1))
        # Feed new image to LSTM
        self.model.save_input([self.x, self.y], img)

    def ros_pose_callback(self, pose_msg):
        """Update internal position state."""
        self.x = pose_msg.position.x
        self.y = pose_msg.position.y

    def ros_publish_waypoint(self, action):
        """Publish single-point ROS trajectory message with given x, y and default z, att."""
        # create trajectory msg
        traj = MultiDOFJointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.header.frame_id = 'frame'
        traj.joint_names.append('base_link')

        # create end point for trajectory
        transforms = Transform()
        transforms.translation.x = action[0]
        transforms.translation.y = action[1]
        transforms.translation.z = Z

        quat = tf.transformations.quaternion_from_euler(0, 0, 0, axes='rzyx')
        transforms.rotation.x = quat[0]
        transforms.rotation.y = quat[1]
        transforms.rotation.z = quat[2]
        transforms.rotation.w = quat[3]

        velocities = Twist()
        accel = Twist()
        point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accel], rospy.Time())
        traj.points.append(point)
        self.waypoint_publisher.publish(traj)

    def wait_for_arrival(self, dest, check_freq=0.05, tol=0.03):
        """
        Wait until drone has arrived at the specified destination x, y.

        :param dest: x, y pair of the destination position
        :param (optional) check_freq : Frequency (Hz) with which to check arrival
        :param (optional) tol: Error tolerance for x and y
        :return: none
        """
        while abs(self.x - 0.132646 - dest[0]) > tol or abs(self.y - dest[1]) > tol:
            time.sleep(check_freq)
        self.ready_for_img = True
