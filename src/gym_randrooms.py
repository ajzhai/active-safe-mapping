import numpy as np
import os
import time
import random
import rospy
import tf
import cv2
import matplotlib.pyplot as plt
import gym
from gym.spaces import Tuple, Box
from sensor_msgs.msg import Image as ImageMsg
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Twist, Transform, Pose
from PIL import Image
import drone_classifier as dc

X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 2.5, -2.5, 2.5
X_START, Y_START = (X_MIN + X_MAX) / 2, (Y_MIN + Y_MAX) / 2
Z = 3.
MIN_WALL_DIST = 0.3
DRONE_X_LENGTH, DRONE_Y_LENGTH = 0.2, 0.2
TABLE_X_LENGTH, TABLE_Y_LENGTH = 1., 1.
TABLES_PER_ROOM = 10
MAV_NAME = 'firefly'
OUTPUT_FILE = '/home/azav/results/same_room_test2.txt'

def make_grid_pool(grid_size):
    """
    Create a pool of room positions in an evenly-spaced grid.

    :param grid_size: The distance between grid points
    :return: List of [x, y] pairs
    """
    pool = []
    for x in np.arange(X_MIN + grid_size / 2., X_MAX - grid_size / 2., grid_size):
        for y in np.arange(Y_MIN + grid_size / 2., Y_MAX - grid_size / 2., grid_size):
            pool.append([x, y])
    return pool

def make_rand_pool(n_points):
    """
    Create a pool of room positions sampled uniformly randomly.

    :param n_points: The size of the desired pool
    :return: List of [x, y] pairs
    """
    pool = np.zeros((n_points, 2))
    pool[:, 0] = np.random.rand(n_points) * (X_MAX - X_MIN) + X_MIN
    pool[:, 1] = np.random.rand(n_points) * (Y_MAX - Y_MIN) + Y_MIN
    return pool


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
        new_x = X_MIN + TABLE_X_LENGTH / 2 + np.random.rand() * (X_MAX - X_MIN - TABLE_X_LENGTH)
        new_y = Y_MIN + TABLE_Y_LENGTH / 2 + np.random.rand() * (Y_MAX - Y_MIN - TABLE_Y_LENGTH)
        while is_overlap(centers, new_x, new_y):
            new_x = X_MIN + TABLE_X_LENGTH / 2 + np.random.rand() * (X_MAX - X_MIN - TABLE_X_LENGTH)
            new_y = Y_MIN + TABLE_Y_LENGTH / 2 + np.random.rand() * (Y_MAX - Y_MIN - TABLE_Y_LENGTH)
        centers.append([new_x, new_y])
    return centers

fixed_table_centers = random_table_centers(TABLES_PER_ROOM)
fixed_pool =  make_rand_pool(10000)

class RandomRooms(gym.Env):
    """
    RL environment for exploration of a set of rooms. Each room gets one
    episode of a fixed number of timesteps. The order of the rooms is
    uniformly random.
    """

    def __init__(self, config):
        rospy.init_node('agent', anonymous=True)
        self.model = dc.Classifier(dc.CNN)
        self.ready_for_img = False
        rospy.Subscriber('/' + MAV_NAME + '/vi_sensor/left/image_raw', ImageMsg,
                         self.ros_img_callback)
        rospy.Subscriber('/' + MAV_NAME + '/ground_truth/pose', Pose,
                         self.ros_pose_callback)
        self.waypoint_publisher = rospy.Publisher('/firefly/command/trajectory',
                                                  MultiDOFJointTrajectory)

        self.x, self.y, self.t = 0., 0., 0.
        self.start_time = time.time()
        self.time_weight = config['time_weight']  # coefficient of time (s) in reward
        self.ep_len = config['ep_length']  # episode length in actual time
        self.map_scale = config['map_scale']  # pixels per unit length
        self.img_interval = config['img_interval']  # how far apart each image is (at most)
        self.true_map = None
        #self._enter_new_room(True)
        #self.ros_publish_waypoint([0, 0])
        self.ep_total_reward = 0.
        time.sleep(3.)

        self.action_space = Box(low=np.array([X_MIN + MIN_WALL_DIST, Y_MIN + MIN_WALL_DIST]), 
                                high=np.array([X_MAX - MIN_WALL_DIST, Y_MAX - MIN_WALL_DIST]),
                                dtype=np.float32)
        self.observation_space = Tuple((Box(low=np.array([X_MIN, Y_MIN]),
                                            high=np.array([X_MAX, Y_MAX]),
                                            dtype=np.float32),
                                        Box(low=-np.inf, high=np.inf,
                                            shape=(len(self.model.latent_state()),),
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
        table_centers = fixed_table_centers # TEMP#####################
        for i in range(TABLES_PER_ROOM):
            os.system("rosrun gazebo_ros spawn_model -database cafe_table -sdf " +
                      "-model table" + str(i) +
                      " -x " + str(table_centers[i][0]) +
                      " -y " + str(table_centers[i][1]))

        # Calculate new true safe map
        self.true_map = np.zeros((int((X_MAX - X_MIN) * self.map_scale), 
                                  int((Y_MAX - Y_MIN) * self.map_scale)))
        safe_x_extent = (TABLE_X_LENGTH - DRONE_X_LENGTH) / 2
        safe_y_extent = (TABLE_Y_LENGTH - DRONE_Y_LENGTH) / 2
        for x, y in table_centers:
            self.true_map[int((x - safe_x_extent) * self.map_scale):
                          int((x + safe_x_extent) * self.map_scale),
                          int((y - safe_y_extent) * self.map_scale):
                          int((y + safe_y_extent) * self.map_scale)] = 1

    def reset(self):
        """Called at the end of each episode(room) to enter a new room and reset position."""
        if not isinstance(self.true_map, type(None)):
            self.save_model_performance()
        self.ros_publish_waypoint([X_START, Y_START])
        self._enter_new_room()
        self.model.clear_image_embeddings()
        self.wait_for_arrival([X_START, Y_START])
        while len(self.model.hybrid_image_embeddings) == 0:
            time.sleep(0.1)
        self.x, self.y, self.t = 0., 0., 0.
        self.ep_total_reward = 0.
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
        print('query: ', action)
        print('label: ', label)
        model_err = self._get_model_improvement(label)
        self.model.train_classifier([self.x, self.y], label)
        print('current room images: ' + str(len(self.model.hybrid_image_embeddings)) + '\n')
        
        self.x, self.y, self.t = 0., 0., 0.
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
        reward = model_err  # - self.time_weight * travel_time
        self.ep_total_reward += reward
        done = self.t >= self.ep_len
        return self.agent_observation(), reward, done, {}

    def _get_label(self, action):
        """Get the true safe map label at the given position."""
        # Converting query position to pixel indices
        x_idx = int((action[0] - X_MIN) * self.map_scale)
        y_idx = int((action[1] - Y_MIN) * self.map_scale)
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
            return self.model.get_loss([self.x, self.y], label)
        elif mode == 'lcu':
            return self.model.uncertainty(label)
        else:
            return 0

    def ros_img_callback(self, img_msg):
        """Send image to safe-mapper LSTM upon ROS image message."""
        if not self.ready_for_img:
            return
        self.ready_for_img = False

        # Convert to 256x256 grayscale array
        img = np.array(Image.frombuffer('RGB', (img_msg.width, img_msg.height),
                                        img_msg.data, 'raw', 'RGB', 0, 1))
        cropped_start_idx = img.shape[1]/2 - img.shape[0]/2  # crop the width
        img = cv2.resize(img[:, cropped_start_idx:cropped_start_idx + img.shape[0]], (256, 256))

        # Feed new image to LSTM
        self.model.encode_input([self.x, self.y], img)

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

    def save_model_performance(self):
        """Calculates average error in current room and writes to file."""
        preds = []
        for pos in fixed_pool:
            preds.append(self._get_label(pos))
        print('safe fraction: %d/%d' % (np.sum(preds), len(preds)))
        loss = self.model.get_batch_loss(fixed_pool, preds)
        conf = np.mean(self.model.get_batch_confidence(fixed_pool))
        acc = self.model.get_batch_accuracy(fixed_pool, preds)
        with open(OUTPUT_FILE, 'a') as f:
            f.write('%f %f %f %f\n' % (acc, loss, conf, self.ep_total_reward))
