#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Header, ColorRGBA
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, Vector3, PointStamped, Pose, PoseStamped, Quaternion, PoseArray
from nav_msgs.msg import Odometry
import tf
import tf.transformations as transform
import math
import time
import numpy as np
from nav_msgs.srv import GetMap
from visualization_msgs.msg import Marker
from threading import Lock


class MCL:
    def __init__(self):
        rospy.init_node('mcl_node', anonymous=True)

        self.MAX_PARTICLES = 40
        self.SKIP_STEP = 30
        self.subsampled_laser_scan = None
        self.map_data = None
        self.free_space = None
        self.state_lock = Lock()
        self.subsampled_laser_angles = None
        self.MIN_RANGE = None
        self.MAX_RANGE = rospy.get_param('~max_range', 5.0)
        self.MAP_X_MAX= 5.0
        self.MAP_X_MIN = 0.0
        self.MAP_Y_MAX = 5.0
        self.MAP_Y_MIN = 0.0
        self.NOISE_X = 0.05  # 0.05
        self.NOISE_Y = 0.05  # 0.05
        self.NOISE_THETA = 0.1  # 0.1
        self.LINEAR_UPDATE = 0.1
        self.ANGULAR_UPDATE = 0.1

        self.linear_delta = 0
        self.angular_delta=0

        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        self.copy_particles = np.copy(self.particles)
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.deltas = np.zeros((self.MAX_PARTICLES, 3))

        self.mean_pose = np.array([0, 0, 0])
        self.current_pose = np.array([0, 0, 0])
        self.last_pose = np.array([0, 0, 0])
        self.odom_data = np.array([0, 0, 0])

        self.odom_initialized = False
        self.lidar_initialized = False
        self.particles_initialized = False
        self.mcl_run_flag = False
        self.rviz_display_flag = True
        self.map_read_flag = False
        self.DEBUG = False

        rospy.Subscriber("scan", LaserScan, self.scan_callback)
        rospy.Subscriber("/odom", Odometry, self.odometry_callback)
        self.cmd_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
        self.pose_pub = rospy.Publisher("/mcl_pose", PoseStamped, queue_size=1)
        self.particles_pub = rospy.Publisher("/particles", PoseArray, queue_size=1)
        self.pub_fake_scan = rospy.Publisher("/fake_scan", LaserScan, queue_size=1)
        self.odom_pub = rospy.Publisher("mcl/odom", Odometry, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.rviz_display_callback)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.read_occupancy_gridmap()
        self.initialize_particles('local',self.particle_to_pose((0,0,0)))
        # self.initialize_particles('global')

    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def normalize(self, x):
        return x / np.linalg.norm(x)

    def quaternion_to_euler_yaw(self, q):
        _, _, yaw = transform.euler_from_quaternion((q.x, q.y, q.z, q.w))
        return yaw

    def rviz_display_callback(self, event):
        if self.rviz_display_flag:
            self.show_particles_in_rviz(self.particles)

    def create_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def particle_to_pose(self, particle):
        pose = Pose()
        pose.position.x = particle[0]
        pose.position.y = particle[1]
        quat_data = transform.quaternion_from_euler(0, 0, particle[2])
        pose.orientation.z = quat_data[2]
        pose.orientation.w = quat_data[3]
        return pose

    def metric_to_grid_coords(self, x, y):
        gx = (x - self.map_data.origin.position.x) / self.map_data.resolution
        gy = (y - self.map_data.origin.position.y) / self.map_data.resolution
        row = min(max(int(gy), 0), self.map_data.height)
        col = min(max(int(gx), 0), self.map_data.width)
        return (row, col)

    def show_particles_in_rviz(self, particles):
        poses_particles = PoseArray()
        poses_particles.header = self.create_header("map")  # should be map
        poses_particles.poses = map(self.particle_to_pose, particles)
        self.particles_pub.publish(poses_particles)

    def odometry_callback(self, msg):
        current_angle = self.quaternion_to_euler_yaw(msg.pose.pose.orientation)
        # print(msg.pose.pose.position.x, msg.pose.pose.position.y,current_angle)
        self.current_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, current_angle])
        if not self.odom_initialized:
            self.last_pose = np.copy(self.current_pose)
            self.odom_initialized = True
        else:
            cos_theta = np.cos(-self.last_pose[2])
            sin_theta = np.sin(-self.last_pose[2])
            rot = np.matrix([[cos_theta, -sin_theta],
                             [sin_theta, cos_theta]])  # rotation matrix to convert global coordinate to local
            delta = np.array(
                [self.current_pose[0:2] - self.last_pose[0:2]]).transpose()  # taking transpose to convert to column vector
            local_delta = (rot * delta).transpose()  # using rotation matrix to get local deltas
            self.odom_data = np.array([local_delta[0, 0], local_delta[0, 1], current_angle - self.last_pose[2]])
            # self.last_pose = np.copy(self.current_pose)
            self.linear_delta = np.linalg.norm(np.array([local_delta[0, 0], local_delta[0, 1]]))
            self.angular_delta = current_angle - self.last_pose[2]
            if self.DEBUG:
                print("odom_callback")
                print "current-pose=",self.current_pose
                print(local_delta)

            # if not self.mcl_run_flag:
            #     # linear_delta = np.linalg.norm(np.array([local_delta[0, 0], local_delta[0, 1]]))
            #     # print "linear_update=", linear_delta
            #     self.mcl_run_flag = True
            #     self.run_MCL()


    def scan_callback(self, msg):
        #filter out nan and inf values as well
        # print("scan_callback")
        self.subsampled_laser_scan = np.array(msg.ranges[::self.SKIP_STEP])
        if not self.lidar_initialized:
            laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.subsampled_laser_angles = laser_angles[::self.SKIP_STEP]
            self.MIN_RANGE = msg.range_min
            if self.MAX_RANGE > msg.range_max:
                self.MAX_RANGE = msg.range_max
            self.lidar_initialized = True
            # print(self.subsampled_laser_angles)

    def publish_scan(self, angles, ranges):
        ls = LaserScan()
        ls.header = self.create_header("base_laser_link")
        ls.angle_min = np.min(angles)
        ls.angle_max = np.max(angles)
        ls.angle_increment = np.abs(angles[0] - angles[1])
        ls.range_min = 0
        ls.range_max = np.max(ranges)
        ls.ranges = ranges
        self.pub_fake_scan.publish(ls)

    def read_occupancy_gridmap(self):
        rospy.wait_for_service("static_map")
        grid_map = rospy.ServiceProxy("static_map", GetMap)().map
        self.map_data = grid_map.info
        self.MAP_X_MAX = self.map_data.width*self.map_data.resolution + self.map_data.origin.position.x
        self.MAP_X_MIN = self.map_data.origin.position.x
        self.MAP_Y_MAX = self.map_data.height*self.map_data.resolution + self.map_data.origin.position.y
        self.MAP_Y_MIN = self.map_data.origin.position.y
        print(grid_map.info)
        # print(grid_map.data)
        grid_matrix = np.array(grid_map.data).reshape((grid_map.info.height, grid_map.info.width))
        print(grid_map.info.height, grid_map.info.width)
        self.free_space = np.zeros_like(grid_matrix)
        self.free_space[grid_matrix == 0] = 1
        # print(self.free_space)
        # print(self.free_space[0,0])0pl
        self.map_read_flag = True

    def initialize_particles(self, mode='global', pose=None):
        print("initialize particles locally around a certain pose or globally about the full map")
        # self.state_lock.acquire()
        if mode == 'global':
            print("global initialization")
            free_y, free_x = np.where(self.free_space == 1)
            indices = np.random.randint(0, len(free_y), size=self.MAX_PARTICLES)
            self.particles[:, 0] = free_x[indices] * self.map_data.resolution + self.map_data.origin.position.x
            self.particles[:, 1] = free_y[indices] * self.map_data.resolution + self.map_data.origin.position.y
            self.particles[:, 2] = 2 * np.pi * np.random.random(self.MAX_PARTICLES)


        elif mode == 'local':
            print("local initialization")
            self.particles[:, 0] = pose.position.x + np.random.normal(loc=0.0, scale=0.5, size=self.MAX_PARTICLES)
            self.particles[:, 1] = pose.position.y + np.random.normal(loc=0.0, scale=0.5, size=self.MAX_PARTICLES)
            self.particles[:, 2] = self.quaternion_to_euler_yaw(pose.orientation) + np.random.normal(loc=0.0, scale=0.4,
                                                                                                     size=self.MAX_PARTICLES)
        self.copy_particles = np.copy(self.particles)
        self.particles_initialized = True
        # self.state_lock.release()

    def compute_motion_model(self, copy_odom):
        print("computing odometry motion model")
        cosines = np.cos(self.copy_particles[:, 2])
        sines = np.sin(self.copy_particles[:, 2])
        self.deltas[:, 0] = cosines * copy_odom[0] - sines * copy_odom[1]
        self.deltas[:, 1] = sines * copy_odom[0] + cosines * copy_odom[1]
        self.deltas[:, 2] = copy_odom[2]
        noise_x = np.random.normal(loc=0.0, scale=self.NOISE_X, size=self.MAX_PARTICLES)
        noise_y = np.random.normal(loc=0.0, scale=self.NOISE_Y, size=self.MAX_PARTICLES)
        noise_theta = np.random.normal(loc=0.0, scale=self.NOISE_THETA, size=self.MAX_PARTICLES)
        self.copy_particles[:, 0] += self.deltas[:, 0] + noise_x
        self.copy_particles[:, 1] += self.deltas[:, 1] + noise_y
        self.copy_particles[:, 2] += self.deltas[:, 2] + noise_theta

    def compute_sensor_model(self):
        print("computing laser sensor model")
        errors = []
        for i in range(self.MAX_PARTICLES):
            errors.append(self.evaluate_sensor_measurement(self.copy_particles[i]))
        # print(errors)
        errors = np.array(errors)
        # errors = np.array(map(self.evaluate_sensor_measurement, self.copy_particles))
        self.weights = np.exp(-errors)
        # print(self.weights)



    def lies_within_map(self,particle):
        if particle[0]< self.MAP_X_MIN or particle[0] > self.MAP_X_MAX:
            return False
        if particle[1] < self.MAP_Y_MIN or particle[1] > self.MAP_Y_MAX:
            return False
        return True


    def evaluate_sensor_measurement(self, particle):
        if not self.lies_within_map(particle):
            print("out of bounds")
            return 300

        row, col = self.metric_to_grid_coords(particle[0], particle[1])
        if not self.free_space[row, col]:
            print("in obstacle region")
            return 300

        local_ranges = self.extract_particle_scan(particle)
        # TODO: compute the difference bwteen predicted ranges and actual ranges
        error = self.subsampled_laser_scan - local_ranges
        normalized_error = np.linalg.norm(error)
        return normalized_error

    def extract_particle_scan(self, particle):
        # also transform base_link pose to laser_pose
        ranges = []
        # print(particle)
        cos_theta = np.cos(particle[2])
        sin_theta = np.sin(particle[2])
        rot = np.matrix([[cos_theta, -sin_theta],
                         [sin_theta, cos_theta]])  # rotation matrix to convert global coordinate to local
        delta = np.array([[0.1, 0.0]]).transpose()  # taking transpose to convert to column vector
        local_delta = (rot * delta).transpose()  # using rotation matrix to get local deltas
        # print( local_delta)
        for angle in np.nditer(self.subsampled_laser_angles):
            phi = particle[2] + angle
            r = self.MIN_RANGE
            while r <= self.MAX_RANGE:
                xm = particle[0] + local_delta[0, 0] + r * np.cos(phi)
                ym = particle[1] + local_delta[0, 1] + r * np.sin(phi)
                row, col = self.metric_to_grid_coords(xm, ym)
                if not self.free_space[row, col]:
                    break
                r += self.map_data.resolution
            ranges.append(r)
        return np.array(ranges)

    def resample_particles(self):
        print("resampling particles")
        resampling_indices = np.random.choice(self.particle_indices, self.MAX_PARTICLES, p=self.weights)
        self.copy_particles = self.particles[resampling_indices, :]

    def publish_odom_and_pose_tf(self, mean_pose=None):
        print("publish odom and pose tf")
        pose_tf = PoseStamped()
        pose_tf.header = self.create_header("map")  # should be map
        pose_tf.pose = self.particle_to_pose(self.mean_pose)
        self.pose_pub.publish(pose_tf)


    def run_MCL(self):
        if self.odom_initialized and self.lidar_initialized and self.particles_initialized:
            total_time= time.time()
            # compute motion model from odometry data
            self.compute_motion_model(np.copy(self.odom_data))


            sensor_model_time = time.time()
            # compute sensor model for laserscan data
            self.compute_sensor_model()
            print "sensor model time= ",time.time()-sensor_model_time
            # normalize importance weights
            self.weights = self.weights / np.sum(self.weights)
            self.particles = np.copy(self.copy_particles)  # probably don't need copy here,,test it

            # resample particles based on the importance weights
            self.resample_particles()

            # find mean pose from all particles
            self.mean_pose = np.dot(self.particles.transpose(), self.weights)

            # publish tranform between base_link and map or between base_footprint and map
            self.publish_odom_and_pose_tf(self.mean_pose)

            # self.show_particles_in_rviz(self.particles)
            self.publish_scan(self.subsampled_laser_angles, self.extract_particle_scan(self.mean_pose))
            print "total_time = ",time.time()-total_time
            self.mcl_run_flag = False

    def test(self):
        # rospy.sleep(2)
        # self.publish_odom_and_pose_tf()
        # self.initialize_particles('local',self.particle_to_pose((2,2,1.57)))
        # self.initialize_particles('global')
        # self.show_particles_in_rviz(self.particles)
        self.publish_scan(self.subsampled_laser_angles, self.extract_particle_scan(np.array([0, 0, 0])))


if __name__ == '__main__':
    particle_filter = MCL()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if not particle_filter.mcl_run_flag:
            if particle_filter.linear_delta!=0 or particle_filter.angular_delta!=0:
                particle_filter.mcl_run_flag = True
                particle_filter.run_MCL()
                particle_filter.last_pose = np.copy(particle_filter.current_pose)
        # # particle_filter.test()
        rate.sleep()
