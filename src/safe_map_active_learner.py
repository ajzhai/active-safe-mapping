#!/home/azhai/anaconda3/envs/py27/bin/python2.7
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image

TRUE_SAFE_MAP_FILE = '/home/azhai/catkin_ws/src/safe_mapping/resource/easy_tables_true_map.png'
MAV_NAME = 'firefly'

def load_true_map(imgfile):
    return cv2.imread(imgfile, 0)

def query_true_map(true_map, x, y):
    x_idx = int(x * 100)
    y_idx = int(y * 100)
    return true_map[x_idx][y_idx]

def imgmsg_to_grayscale_array(img_msg):
    return np.array(Image.frombuffer('RGB', (img_msg.width, img_msg.height), img_msg.data, 'raw', 'L', 0, 1))

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + " I saw something")
    img = imgmsg_to_grayscale_array(data)
    # Do whatever we need to do with img

if __name__ == '__main__':
    rospy.init_node('active_learner', anonymous=True)
    rospy.sleep(10.)  # Waiting for setup to complete
    rospy.Subscriber('/' + MAV_NAME + '/vi_sensor/left/image_raw', ImageMsg, callback)
    rospy.spin()
