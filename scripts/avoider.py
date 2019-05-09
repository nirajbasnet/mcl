#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Header, ColorRGBA
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, Vector3, PointStamped, Pose,PoseStamped,Quaternion
from nav_msgs.msg import Odometry
import tf.transformations as transform
import math
import numpy as np
from visualization_msgs.msg import Marker


class PotentialFieldPlanner:
    def __init__(self):
        rospy.init_node('potential_field_node', anonymous=True)
        rospy.Subscriber("scan", LaserScan, self.scan_callback)
        rospy.Subscriber("/clicked_goal", PoseStamped, self.goal_callback)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.odometry_callback)
        self.cmd_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
        self.goal_marker_pub = rospy.Publisher('goal_marker', Marker, queue_size=10)
        self.pot_marker_pub = rospy.Publisher('potential_marker', Marker, queue_size=10)
        rospy.Timer(rospy.Duration(0.2), self.rviz_display_callback)
        self.publish_flag = False
        self.Kp_att = 2.0
        self.Kp_repul = 0.09
        self.obs_effect_dist = 1.5
        self.lookahead_distance = 1.5
        self.goal = np.array([0, 0, 0])
        self.goal_orientation = 0.0
        self.goal_tolerance = 0.1
        self.step_size = 0.5
        self.current_pos = np.array([0, 0, 0])
        self.current_orientation = 0.0
        self.laserscan_data = LaserScan()
        self.linear_vel=0.12
        self.angular_vel=0
        self.potential_field_direction=0
        self.last_angular_vel=0
        self.rviz_display_flag = False

    def rviz_display_callback(self, event):
        if self.rviz_display_flag:
            self.show_in_rviz()

    def show_in_rviz(self):
        goal_quaternion=transform.quaternion_from_euler(0, 0, self.goal_orientation)
        goal_marker = Marker()
        goal_marker.type=Marker.ARROW
        goal_marker.id=0
        goal_marker.lifetime=rospy.Duration(0.2)
        goal_marker.pose.position=Point(self.goal[0],self.goal[1],self.goal[2])
        goal_marker.pose.orientation.z=goal_quaternion[2]
        goal_marker.pose.orientation.w=goal_quaternion[3]
        # print("position=",goal_marker.pose.position)
        goal_marker.scale=Vector3(0.56, 0.06, 0.06)
        goal_marker.header=Header(frame_id='map')
        goal_marker.color=ColorRGBA(0.0, 1.0, 0.0, 0.8)

        pot_quaternion = transform.quaternion_from_euler(0, 0, self.potential_field_direction)
        pot_marker = Marker()
        pot_marker.type = Marker.ARROW
        pot_marker.id = 0
        pot_marker.lifetime = rospy.Duration(0.2)
        pot_marker.pose.position = Point(self.current_pos[0], self.current_pos[1], self.current_pos[2])
        pot_marker.pose.orientation.z = pot_quaternion[2]
        pot_marker.pose.orientation.w = pot_quaternion[3]
        # print("position=",pot_marker.pose.position)
        pot_marker.scale = Vector3(0.85, 0.06, 0.06)
        pot_marker.header = Header(frame_id='map')
        pot_marker.color = ColorRGBA(0.0, 0.0, 0.0, 0.8)
        self.pot_marker_pub.publish(pot_marker)
        self.goal_marker_pub.publish(goal_marker)


    def goal_callback(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quaternion_data = msg.pose.orientation
        q = (quaternion_data.x, quaternion_data.y, quaternion_data.z, quaternion_data.w)
        self.goal_orientation = transform.euler_from_quaternion(q)[2]
        print(self.goal)
        self.rviz_display_flag = True
        self.publish_flag = True

    def odometry_callback(self, msg):
        odom_pos = msg.pose.pose.position
        quaternion_data = msg.pose.pose.orientation
        self.current_pos = np.array([odom_pos.x, odom_pos.y, 0])
        # print("pos= ",self.current_pos)
        q = (quaternion_data.x, quaternion_data.y, quaternion_data.z, quaternion_data.w)
        self.current_orientation = transform.euler_from_quaternion(q)[2]  # angle is in [-pi, pi]
        # print("orientation= ",self.current_orientation)

    def toXYZ(self, radius, theta):
        return np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0])

    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def normalize(self, x):
        return x / np.linalg.norm(x)

    def calculate_goal_heading(self, target, source):
        return math.atan2(target[1] - source[1], target[0] - source[0])

    def calc_attractive_force(self):
        f_att = self.Kp_att * (self.goal - self.current_pos)
        # print("f-attractive=", f_att)
        return f_att

    def calc_repulsive_force(self):
        f_rep = np.array([0.0, 0.0, 0.0])
        scan_data = self.laserscan_data.ranges
        angle_min = self.laserscan_data.angle_min
        current_angle = angle_min
        for dist in scan_data:
            if dist < self.lookahead_distance:
                obs_point = self.current_pos + self.toXYZ(dist, current_angle + self.current_orientation)

                distance_vector = self.current_pos - obs_point
                obs_source_dist = self.distance(obs_point, self.current_pos)
                rep_force =self.Kp_repul* self.normalize(distance_vector) * ((1.0 / obs_source_dist) - (1.0 / self.obs_effect_dist)) *(1.0/ (obs_source_dist * obs_source_dist))
                f_rep = f_rep + rep_force
            current_angle = current_angle + self.laserscan_data.angle_increment
        # print("f-repulsive=", f_rep)
        return f_rep

    def do_potential_field_planning(self):
        print("planning started")
        plan_rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.distance(self.current_pos, self.goal) < self.goal_tolerance:
                while abs(self.goal_orientation-self.current_orientation) >= math.radians(2):
                    self.publish_twist_vel(0, math.radians(-20))
                    self.rviz_display_flag = False
                print("Planning is finished")
                return
            motion_vec = self.calc_attractive_force() + self.calc_repulsive_force()
            self.potential_field_direction = math.atan2(motion_vec[1], motion_vec[0])
            self.angular_vel = self.step_size*(self.potential_field_direction - self.current_orientation)
            if abs(self.angular_vel-self.last_angular_vel)>=0.8:
                self.angular_vel = (self.angular_vel + self.last_angular_vel)/4.0
                self.last_angular_vel= self.angular_vel
            print("motion_vel=", self.angular_vel)
            self.publish_twist_vel(self.linear_vel,self.angular_vel)
            plan_rate.sleep()
            print(motion_vec)

    def publish_twist_vel(self, vel, omega):
        command = Twist()
        command.linear.x = vel
        command.angular.z = omega
        self.cmd_pub.publish(command)

    def scan_callback(self, msg):
        self.laserscan_data = msg


if __name__ == '__main__':
    pot_field = PotentialFieldPlanner()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if pot_field.publish_flag:
            pot_field.do_potential_field_planning()
            print("publish_data")
            pot_field.publish_flag = False
        rate.sleep()
