#!/usr/bin/env python3

import rospy

# import sys
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Point
from ford_msgs.msg import Clusters
from visualization_msgs.msg import Marker, MarkerArray
from crowd_obs import Observation_Crowd
import numpy as np
import math
from nav_msgs.msg import Odometry
import configparser
import torch
import os
import copy
from crowd_nav.configs.config_dsrnn import Config 
from pytorchBaselines.a2c_ppo_acktr.model import Policy

from std_msgs.msg import Bool


PED_RADIUS = 0.3
# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff

class NN_tb3:
    def __init__(self,policy_config,  env_config, policy, ns,device):

        #
        self.ns = ns
        self.ns_prefix = lambda x: os.path.join(self.ns, x)
        self.env_config = env_config
        self.device = device
        self.policy = policy
        self.policy_config = policy_config
        # for action
        self.angle2Action = 0
        self.distance = 0
        self.other_agents_state = {}

        # for subscribers
        self.pose = PoseStamped()
        self.vel = Vector3()
        self.psi = 0.0
        self.num_poses = 0
        self.fixed_time_interval = 0.1
        # for publishers
        self.global_goal = PoseStamped()
        self.goal = PoseStamped()
        self.desired_position = PoseStamped()
        self.desired_action = np.zeros((2,))

        # # publishers
        self.pub_twist = rospy.Publisher(self.ns_prefix("cmd_vel"), Twist, queue_size=1)
        self.pub_path_marker = rospy.Publisher(self.ns_prefix("action"), Marker, queue_size=1)
        self.observation = Observation_Crowd(ns=self.ns)
        self.pub_agent_markers = rospy.Publisher(self.ns_prefix('other_agents_markers'),MarkerArray,queue_size=1)
        self.pub_vis = rospy.Publisher(self.ns_prefix("obs_range"),MarkerArray,queue_size=1)
        self.pub_shutdown = rospy.Publisher("record_shutdown",Bool,queue_size=1)
        # # sub
        self.sub_pose = rospy.Subscriber(self.ns_prefix("odom"), Odometry, self.cbPose)
        self.sub_global_goal = rospy.Subscriber(self.ns_prefix("move_base_simple/goal"), PoseStamped, self.cbGlobalGoal)
        self.sub_subgoal = rospy.Subscriber(self.ns_prefix("subgoal"), PoseStamped, self.cbSubGoal)
        self.sub_reset = rospy.Subscriber(self.ns_prefix("reset"), Bool, self.cbREset)
        self.sub_clusters = rospy.Subscriber(self.ns_prefix("crowd_obs"), Clusters, self.cbClusters)


        self.last_left = 0.
        self.last_right = 0.
        self.desired_reset_num = int(rospy.get_param("/desired_resets",1))
        self.improved_action = bool(rospy.get_param("/imporved_action",False))
        print(f"[improved_action]:{self.improved_action}")
        # subgoals
        self.sub_goal = Vector3()
        # get crowd:
        self.reset_num = 0
        # control timer
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.cbControl)
        self.nn_timer = rospy.Timer(rospy.Duration(0.1), self.cbComputeActionCrowdNav)
        self.eval_recurrent_hidden_states = {}
        self.last_v = 0.0
        self.last_w = 0.0
        node_num = 1
        edge_num = self.policy.base.human_num + 1
        self.eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(1, node_num, self.policy_config.SRNN.human_node_rnn_size * 1,
                                                                    device=self.device)

        self.eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(1, edge_num,
                                                                        self.policy_config.SRNN.human_human_edge_rnn_size*1,
                                                                        device=self.device)

        self.eval_masks = torch.zeros(1, 1, device=self.device)
        self.new_global_goal_received = False
        self.robot_vx = 0
        self.robot_vy = 0
    



    def vis_ob_range(self):
        marker = Marker()
        r, g, b, a = [0.9, 0.1, 0.1, 0.1]
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "test_1/map"
        marker.ns = "scan_range"
        marker.action = Marker.MODIFY
        marker.type = Marker.SPHERE
        marker.scale = Vector3(5.0*2,5.0*2,0.1)
        marker.color = ColorRGBA(r, g, b, a)
        marker.lifetime = rospy.Duration(1)
        marker.pose.position.x = self.pose.pose.position.x
        marker.pose.position.y = self.pose.pose.position.y
        self.pub_vis.publish([marker])

    def cbREset(self,msg):
        is_shut = Bool()



        if msg.data == True:
            self.reset_num += 1
            self.num_poses = 0
            
            node_num = 1
            edge_num = self.policy.base.human_num + 1
            self.eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(1, node_num, 
                                                                              self.policy_config.SRNN.human_node_rnn_size * 1,
                                                                        device=self.device)

            self.eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(1, edge_num,
                                                                            self.policy_config.SRNN.human_human_edge_rnn_size*1,
                                                                            device=self.device)

            self.eval_masks = torch.zeros(1, 1, device=self.device)
            self.last_left  = 0.
            self.last_right = 0.
            self.desired_action = np.zeros((2,))
            self.last_v = 0.0
            self.last_w = 0.0
            self.robot_vx = 0
            self.robot_vy = 0

        if self.desired_reset_num- self.reset_num == 0:
            is_shut.data= True
            self.pub_shutdown.publish(is_shut.data)
            rospy.signal_shutdown("Everything is done")
        
    


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
        self.new_global_goal_received = True
        self.global_goal = msg
        self.goal.pose.position.x = msg.pose.position.x
        self.goal.pose.position.y = msg.pose.position.y
        self.goal.header = msg.header


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
        self.num_poses += 1
        self.vis_ob_range()
        self.cbVel(msg)
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
        rospy.logwarn(self.distance)

    def cbVel(self, msg):
        self.vel = msg.twist.twist.linear

    def cbClusters(self,msg):
        xs = []
        ys = []
        radii = []
        labels = []
        num_clusters = len(msg.mean_points)
        vx = []
        vy = []
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
        self.visualize_other_agents(xs, ys, radii, labels)

    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)

    def update_action(self, action):
        self.desired_action = action
        self.desired_position.pose.position.x = self.pose.pose.position.x + (action[0])
        self.desired_position.pose.position.y = self.pose.pose.position.y + (action[1])

    def max_yaw(self, a):
        phi = 0
        phi_tol = 0.1

        if 0 < a <= phi_tol:
            phi = -0.2
        if phi_tol < a <= 3*phi_tol:
            phi = -0.3
        if 3*phi_tol < a < 7*phi_tol:
            phi = -0.4
        if a >= 7*phi_tol:
            phi = -0.5
        # neg
        if 0 > a >= -phi_tol:
            phi = 0.2
        if -phi_tol > a >= -3*phi_tol:
            phi = 0.3
        if -3*phi_tol > a > -7*phi_tol:
            phi = 0.4
        if a <= -7*phi_tol:
            phi = 0.5

        return phi
    
    def cbControl(self, event):

        twist = Twist()
        if self.new_global_goal_received:
            if not self.goalReached():
                if self.improved_action:
                    act_norm = np.linalg.norm(self.desired_action)
                    if act_norm > 1.0:
                        self.desired_action[0] = self.desired_action[0] / act_norm * 1.0
                        self.desired_action[1] = self.desired_action[1] / act_norm * 1.0
                    vel = np.array([self.desired_action[0], self.desired_action[1]])

                    if abs(self.angle2Action) < math.pi/2:
                        twist.linear.x = 0.3*np.linalg.norm(vel)
                    else:
                        twist.linear.x = 0.1*np.linalg.norm(vel)

                    twist.angular.z = self.max_yaw(self.angle2Action)

                else:
                    if abs(self.angle2Action) > 0.1 and self.angle2Action > 0:
                        twist.angular.z = -0.3
                    elif abs(self.angle2Action) > 0.1 and self.angle2Action < 0:
                        twist.angular.z = 0.3
                    vel = np.array([self.desired_action[0], self.desired_action[1]])
                    twist.linear.x = 0.2 * np.linalg.norm(vel)

        self.pub_twist.publish(twist)

    def cbComputeActionCrowdNav(self, event):
        robot_x = self.pose.pose.position.x
        robot_y = self.pose.pose.position.y
        # goal
        goal_x = self.sub_goal.x
        goal_y = self.sub_goal.y

        robot_vx = self.vel.x
        robot_vy = self.vel.y
        # oriantation
        theta = self.psi
        robot_radius = 0.3

        # set robot info
        ob = {}

        ob['robot_node'] = torch.tensor([[[robot_x, robot_y, robot_radius, goal_x, goal_y, 0.3, theta]]],dtype = torch.float32,device=self.device)
        ob['temporal_edges'] = torch.tensor([[[robot_vx, robot_vy]]],dtype = torch.float32,device=self.device)
        obstacle_x = torch.tensor([-6.0, -6.0, -6.0, -6.0, -6.0])
        obstacle_y = torch.tensor([-6.0, -6.0, -6.0, -6.0, -6.0])
        # velocity
        obstacle_vx = [0.0, 0.0, 0.0, 0.0, 0.0]
        obstacle_vy = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        humans = self.env_config.getint("sim", "human_num")
        ob['spatial_edges'] = torch.zeros((1,humans, 2),device=self.device)
        if len(self.other_agents_state) > 0:
            
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
        else:
            if "v" in self.other_agents_state:
                v = self.other_agents_state["v"]
                for i in range(len(v[0])):
                    vm = math.sqrt(v[0][i] ** 2 + v[1][i] ** 2)

        for i in range(humans):
            relative_pos = torch.tensor([[[obstacle_x[i] - robot_x, obstacle_y[i] - robot_y]]],device=self.device)
            ob['spatial_edges'][0][i] = relative_pos

        _, action, _, self.eval_recurrent_hidden_states = self.policy.act(ob,
                                                                          self.eval_recurrent_hidden_states,
                                                                        self.eval_masks,
                                                                        deterministic=True)
        self.eval_masks = torch.tensor([[1.0]],dtype=torch.float32,device=self.device)
        self.update_action(action[0].detach().numpy())
        self.update_angle2Action()

    def update_subgoal(self, subgoal):
        self.goal.pose.position.x = subgoal[0]
        self.goal.pose.position.y = subgoal[1]

    def visualize_path(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "test_1/map"
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

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "test_1/map"
        marker.ns = 'path_trail'
        marker.id = self.num_poses 
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position = copy.deepcopy(self.pose.pose.position)
        marker.scale = Vector3(x=0.2,y=0.2,z=0.2)
        marker.color = ColorRGBA(g=0.0,r=0,b=1.0,a=0.3)
        marker.lifetime = rospy.Duration(60)
        self.pub_path_marker.publish(marker)

    def visualize_other_agents(self,xs,ys,radii,labels):
        markers = MarkerArray()
        markers.markers.clear()
        for i in range(len(xs)):
            if xs[i] !=0:
                marker = Marker()
                marker.header.stamp = rospy.Time.now()
                marker.header.frame_id = "test_1/map"
                marker.ns = 'other_agent'
                marker.id = labels[i]
                marker.type = marker.CYLINDER
                marker.action = marker.ADD
                marker.pose.position.x = xs[i]
                marker.pose.position.y = ys[i]
                marker.scale = Vector3(x=2*0.2,y=2*0.2,z=1)
                marker.color = ColorRGBA(r=1.0,g=0.4,a=0.3)
                marker.lifetime = rospy.Duration(1)
                markers.markers.append(marker)

        self.pub_agent_markers.publish(markers)

    def on_shutdown(self):
        rospy.loginfo("[%s] Shutting down.")
        self.stop_moving()
        rospy.loginfo("Stopped %s's velocity.")

import gym

def run():
    policy_name = str(rospy.get_param("/training_model","lstm")).lower()
    print(f"[policy_name]:{policy_name}")
    device = "cpu"
    # the path of training result which contains configs and rl mode
    path_current_directory = os.path.dirname(__file__)

    config = Config()

    
    env_config_file = os.path.join(
        path_current_directory, "crowd_nav/data/output/", "env.config"
    )  # path beginging without slash

    model_weights = os.path.join(
        path_current_directory, "crowd_nav/data/output/", f"rl_model_{policy_name}.pth"
    )

    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    d={}
    # robot node: px, py, r, gx, gy, v_pref, theta
    d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,7,), dtype = np.float32)
    # only consider the robot temporal edge and spatial edges pointing from robot to each human
    d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,2,), dtype=np.float32)
    d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, 2), dtype=np.float32)
    observation_space=gym.spaces.Dict(d)

    high = np.inf * np.ones([2, ])
    action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    actor_critic = Policy(observation_space.spaces,  # pass the Dict into policy to parse
		action_space,
		base_kwargs=config,
		base=config.robot.policy)

    actor_critic.load_state_dict(torch.load(model_weights,map_location=device))
    policy_config = config
    rospy.init_node("crowdnav_dsrnn", anonymous=False)
    print("==================================\ncrowdnav node started")
    ns = "/test_1/"
    nn_tb3 = NN_tb3( policy_config,env_config, actor_critic, ns,device)
    rospy.on_shutdown(nn_tb3.on_shutdown)
    rospy.spin()


if __name__ == "__main__":
    run()
