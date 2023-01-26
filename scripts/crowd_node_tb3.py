#!/usr/bin/env python3

import rospy

# import sys
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Point
from ford_msgs.msg import Clusters
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import math
from nav_msgs.msg import Odometry
import configparser
import torch
import gym
import os

from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.crowd_sim import CrowdSim


PED_RADIUS = 0.3
# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff


class NN_tb3:
    def __init__(self, env, env_config, policy):

        #
        self.env = env
        self.env_config = env_config
        # configure robot
        self.robot = Robot(env_config, "robot")
        self.robot.set_policy(policy)

        self.env.set_robot(self.robot)  # pass robot parameters into env
        self.ob = env.reset(
            "test", 1
        )  # intial some parameters from .config file such as time_step,success_reward for other instances
        self.policy = policy
        self.policy.set_env(env)
        # for action
        self.angle2Action = 0
        self.distance = 0
        self.other_agents_state = {}

        # for subscribers
        self.pose = PoseStamped()
        self.vel = Vector3()
        self.psi = 0.0

        # for publishers
        self.global_goal = PoseStamped()
        self.goal = PoseStamped()
        self.desired_position = PoseStamped()
        self.desired_action = np.zeros((2,))

        # # publishers
        self.pub_twist = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        # self.pub_pose_marker = rospy.Publisher('',Marker,queue_size=1)
        # self.pub_agent_markers = rospy.Publisher('~agent_markers',MarkerArray,queue_size=1)
        self.pub_path_marker = rospy.Publisher("action", Marker, queue_size=1)
        # self.pub_goal_path_marker = rospy.Publisher('~goal_path_marker',Marker,queue_size=1)
        # # sub
        self.sub_pose = rospy.Subscriber("odom", Odometry, self.cbPose)
        self.sub_global_goal = rospy.Subscriber("goal", PoseStamped, self.cbGlobalGoal)
        self.sub_subgoal = rospy.Subscriber("subgoal", PoseStamped, self.cbSubGoal)

        # subgoals
        self.sub_goal = Vector3()

        self.sub_clusters = rospy.Subscriber("~clusters", Clusters, self.cbClusters)

        # control timer
        self.control_timer = rospy.Timer(rospy.Duration(0.2), self.cbControl)
        self.nn_timer = rospy.Timer(rospy.Duration(0.01), self.cbComputeActionCrowdNav)

    def update_angle2Action(self):
        # action vector
        v_a = np.array(
            [
                self.desired_position.pose.position.x - self.pose.pose.position.x,
                self.desired_position.pose.position.y - self.pose.pose.position.y,
            ]
        )
        # pose direction
        e_dir = np.array([math.cos(self.psi), math.sin(self.psi)])
        # angle: <v_a, e_dir>
        self.angle2Action = np.math.atan2(
            np.linalg.det([v_a, e_dir]), np.dot(v_a, e_dir)
        )

    def cbGlobalGoal(self, msg):
        self.stop_moving_flag = True
        self.new_global_goal_received = True
        self.global_goal = msg
        self.goal.pose.position.x = msg.pose.position.x
        self.goal.pose.position.y = msg.pose.position.y
        self.goal.header = msg.header

        # reset subgoals
        # print(
        #     "new goal: " + str([self.goal.pose.position.x, self.goal.pose.position.y])
        # )

    def cbSubGoal(self, msg):
        # update subGoal
        self.sub_goal.x = msg.pose.position.x
        self.sub_goal.y = msg.pose.position.y

    def goalReached(self):
        # check if near to global goal
        if self.distance > 0.3:
            return False
        else:
            return True

    def cbPose(self, msg):
        # update robot vel (vx,vy)
        self.cbVel(msg)
        # get pose angle
        q = msg.pose.pose.orientation
        self.psi = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
        )  # bounded by [-pi, pi]
        self.pose = msg.pose
        self.visualize_path()

        v_p = msg.pose.pose.position
        v_g = self.sub_goal
        v_pg = np.array([v_g.x - v_p.x, v_g.y - v_p.y])
        self.distance = np.linalg.norm(v_pg)
        # self.visualize_pose(msg.pose.pose.position,msg.pose.pose.orientation)

    def cbVel(self, msg):
        self.vel = msg.twist.twist.linear

    def cbClusters(self, msg):
        xs = []
        ys = []
        radii = []
        labels = []
        num_clusters = len(msg.mean_points)
        vx = []
        vy = []
        # print(num_clusters)
        for i in range(num_clusters):
            index = msg.labels[i]
            # if index > 24: #for static map
            x = msg.mean_points[i].x
            y = msg.mean_points[i].y
            v_x = msg.velocities[i].x
            v_y = msg.velocities[i].y
            vx.append(v_x)
            vy.append(v_y)

            inflation_factor = 1.5

            radius = msg.mean_points[i].z * inflation_factor

            xs.append(x)
            ys.append(y)
            radii.append(radius)
            labels.append(index)
        if len(labels) > 0:
            self.other_agents_state["pos"] = [xs, ys]
            self.other_agents_state["v"] = [vx, vy]
            self.other_agents_state["r"] = radii

    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)

    def update_action(self, action):
        # print 'update action'
        self.desired_action = action
        # self.desired_position.pose.position.x = self.pose.pose.position.x + 1*action[0]*np.cos(action[1])
        # self.desired_position.pose.position.y = self.pose.pose.position.y + 1*action[0]*np.sin(action[1])
        self.desired_position.pose.position.x = self.pose.pose.position.x + (action[0])
        self.desired_position.pose.position.y = self.pose.pose.position.y + (action[1])
        # print(action[0])

    def cbControl(self, event):

        twist = Twist()
        if not self.goalReached():
            if abs(self.angle2Action) > 0.1 and self.angle2Action > 0:
                twist.angular.z = -0.3
                # print("spinning in place +")
            elif abs(self.angle2Action) > 0.1 and self.angle2Action < 0:
                twist.angular.z = 0.3
                # print("spinning in place -")
            # else:
            vel = np.array([self.desired_action[0], self.desired_action[1]])
            twist.linear.x = 0.1 * np.linalg.norm(vel)
        self.pub_twist.publish(twist)

    def cbComputeActionCrowdNav(self, event):
        robot_x = self.pose.pose.position.x
        robot_y = self.pose.pose.position.y
        # goal
        goal_x = self.sub_goal.x
        goal_y = self.sub_goal.y
        # velocity
        robot_vx = self.vel.x
        robot_vy = self.vel.y
        # oriantation
        theta = self.psi
        robot_radius = 0.3

        # set robot info
        self.robot.set(
            robot_x, robot_y, goal_x, goal_y, robot_vx, robot_vy, theta, robot_radius
        )

        # obstacle: position, velocity, radius
        # position
        # obstacle_x = [0.1,0.2,0.3,0.4,0.5]
        # obstacle_y = [0.1,0.2,0.3,0.4,0.5]
        # # velocity
        # obstacle_vx = [0.1,0.2,0.3,0.4,0.5]
        # obstacle_vy = [0.1,0.2,0.3,0.4,0.5]

        obstacle_x = [-6.0, -6.0, -6.0, -6.0, -6.0]
        obstacle_y = [-6.0, -6.0, -6.0, -6.0, -6.0]
        # velocity
        obstacle_vx = [0.0, 0.0, 0.0, 0.0, 0.0]
        obstacle_vy = [0.0, 0.0, 0.0, 0.0, 0.0]
        obstacle_radius = 0.3

        humans = self.env_config.getint("sim", "human_num")
        if len(self.other_agents_state) > 0:
            # print(self.other_agents_state)
            obstacle_x = [0.0 for _ in range(humans)]
            obstacle_y = [0.0 for _ in range(humans)]
            obstacle_vx = [0.0 for _ in range(humans)]
            obstacle_vy = [0.0 for _ in range(humans)]

            for prop in self.other_agents_state:
                if prop == "pos":
                    x_y = self.other_agents_state[prop]
                    _it = np.minimum(humans, len(x_y[0]))

                    for i in range(_it):
                        obstacle_x[i] = x_y[0][i]
                        obstacle_y[i] = x_y[1][i]
                if prop == "v":
                    v = self.other_agents_state[prop]
                    for i in range(_it):
                        obstacle_vx[i] = v[0][i]
                        obstacle_vy[i] = v[1][i]
            # print(math.sqrt(obstacle_vx[3]**2+obstacle_vy[3]**2))
            # print("------------------------")
            # for i in range(5):
            #     print("pos", i, ":", obstacle_x[i], " ", obstacle_y[i])
            #     v = math.sqrt(obstacle_vx[i] ** 2 + obstacle_vy[i] ** 2)
            #     print("vel", i, ":", obstacle_vx[i], " ", obstacle_vy[i], "|  v = ", v)
        else:
            if "v" in self.other_agents_state:
                v = self.other_agents_state["v"]
                for i in range(len(v[0])):
                    vm = math.sqrt(v[0][i] ** 2 + v[1][i] ** 2)
                    # if vm > 0:
                    #     print(i, vm)

        # initial obstacle instances and set value
        for i in range(humans):
            self.env.humans[i].set(
                obstacle_x[i],
                obstacle_y[i],
                goal_x,
                goal_y,
                obstacle_vx[i],
                obstacle_vy[i],
                theta,
                obstacle_radius,
            )
            self.ob[i] = self.env.humans[i].get_observable_state()

        # ************************************ Output ************************************
        # get action info
        action = self.robot.act(self.ob)

        # print('\n---------\nrobot position (X,Y):', position.position)
        # print(action)
        # print(theta)

        self.update_action(action)
        self.update_angle2Action()

    def update_subgoal(self, subgoal):
        self.goal.pose.position.x = subgoal[0]
        self.goal.pose.position.y = subgoal[1]

    def visualize_path(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "path_arrow"
        marker.id = 0
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.points.append(self.pose.pose.position)
        marker.points.append(self.desired_position.pose.position)
        marker.scale = Vector3(x=0.1, y=0.2, z=0.2)
        marker.color = ColorRGBA(b=1.0, a=1.0)
        marker.lifetime = rospy.Duration(1)
        self.pub_path_marker.publish(marker)

        # # Display BLUE DOT at NN desired position
        # marker = Marker()
        # marker.header.stamp = rospy.Time.now()
        # marker.header.frame_id = 'map'
        # marker.ns = 'path_trail'
        # marker.id = self.num_poses
        # marker.type = marker.CUBE
        # marker.action = marker.ADD
        # marker.pose.position = copy.deepcopy(self.pose.pose.position)
        # marker.scale = Vector3(x=0.2,y=0.2,z=0.2)
        # marker.color = ColorRGBA(g=0.0,r=0,b=1.0,a=0.3)
        # marker.lifetime = rospy.Duration(60)
        # self.pub_path_marker.publish(marker)

    def on_shutdown(self):
        rospy.loginfo("[%s] Shutting down.")
        self.stop_moving()
        rospy.loginfo("Stopped %s's velocity.")


def run():

    policy_name = "lstm"

    device = "cpu"
    phase = "test"

    select_policy = {"cadrl": CADRL(), "lstm": LstmRL(), "sarl": SARL()}
    # the path of training result which contains configs and rl mode
    path_current_directory = os.path.dirname(__file__)

    env_config_file = os.path.join(
        path_current_directory, "crowd_nav/data/output/", "env.config"
    )  # path beginging without slash
    policy_config_file = os.path.join(
        path_current_directory, "crowd_nav/data/output/", "policy.config"
    )
    model_weights = os.path.join(
        path_current_directory, "crowd_nav/data/output/", f"rl_model_{policy_name}.pth"
    )
    # print(model_weights)
    # select policy
    policy = select_policy[policy_name]  # {SARL(),CADRL(),LstmRL()}
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    policy.get_model().load_state_dict(torch.load(model_weights))
    policy.set_device(device)
    policy.set_phase(phase)

    # configure environment / obstacles
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = CrowdSim()  # env is inherited from CrowdSim class in crowd_sim.py
    env.configure(env_config)

    rospy.init_node("crowdnav_tb3", anonymous=False)
    print("==================================\ncrowdnav node started")

    nn_tb3 = NN_tb3(env, env_config, policy)
    rospy.on_shutdown(nn_tb3.on_shutdown)

    rospy.spin()


if __name__ == "__main__":
    run()
