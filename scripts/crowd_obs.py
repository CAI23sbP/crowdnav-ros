#!/usr/bin/env python3

from ford_msgs.msg import Clusters
from geometry_msgs.msg import Vector3, Point
import rospy 
from pedsim_msgs.msg import AgentStates
import os
import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray,Marker
from std_msgs.msg import ColorRGBA

class Observation_Crowd(object):
    def __init__(self,ns):
        self.ns = ns
        self.cluster = Clusters()
        self.ns_prefix = lambda x: os.path.join(self.ns, x)
        self.pub_cluster = rospy.Publisher(self.ns_prefix("crowd_obs"),Clusters,queue_size=1)
        self.is_pomdp = rospy.get_param("/is_pomdp",False)
        # if not self.is_pomdp:
        self.sub_peds = rospy.Subscriber(self.ns_prefix("pedsim_simulator/simulated_agents"),
                            AgentStates,
                            self.cb_pedsim_data )
        self.sub_odom  = rospy.Subscriber(self.ns_prefix("odom"),
                            Odometry,
                            self.cb_odom )
        # else:
        #     self.sub_peds = rospy.Subscriber(self.ns_prefix("leg_tracker"),
        #                         AgentStates,
        #                         self.cb_pedsim_data )
        self.X = 0

        self.Y = 0

    def sub_header(self,msg):
        self.header=msg.header

    def cb_odom (self,msg):
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y


    def cb_pedsim_data(self,msg):
        crowd = msg.agent_states
        self.cluster = Clusters()
        if  not self.is_pomdp:
            for human in crowd:
                tmp_point = Point()
                tmp_vel = Vector3()
                tmp_id = human.id
                tmp_point.x = human.pose.position.x
                tmp_point.y = human.pose.position.y

                tmp_vel.x = human.twist.linear.x 
                tmp_vel.y = human.twist.linear.x 
                self.cluster.mean_points.append(tmp_point)
                self.cluster.velocities.append(tmp_point)
                self.cluster.labels.append(tmp_id)
        else:

            for human in crowd:
                tmp_point = Point()
                tmp_vel = Vector3()
                tmp_point.x = human.pose.position.x
                tmp_point.y = human.pose.position.y
                dx = tmp_point.x - self.X
                dy = tmp_point.y - self.Y
                dist = np.linalg.norm([dx,dy])
                if dist < 5.0:
                    tmp_id = human.id
                    tmp_point.x = human.pose.position.x
                    tmp_point.y = human.pose.position.y
                    tmp_vel.x = human.twist.linear.x 
                    tmp_vel.y = human.twist.linear.x 
                    self.cluster.mean_points.append(tmp_point)
                    self.cluster.velocities.append(tmp_point)
                    self.cluster.labels.append(tmp_id)
        self.pub_cluster.publish(self.cluster)
