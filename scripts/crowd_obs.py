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
from obstacle_detector.msg import Obstacles
from geometry_msgs.msg import  PoseArray
from message_filters import ApproximateTimeSynchronizer ,Subscriber
import tf
from std_msgs.msg import Int16
class Observation_Crowd(object):
    def __init__(self,ns):
        self.ns = ns
        self.cluster = Clusters()
        self.ns_prefix = lambda x: os.path.join(self.ns, x)
        self.pub_cluster = rospy.Publisher(self.ns_prefix("crowd_obs"),Clusters,queue_size=1)
        self.is_pomdp = rospy.get_param("/is_pomdp",False)
        
        if not self.is_pomdp:
            self.sub_peds = rospy.Subscriber(self.ns_prefix("pedsim_simulator/simulated_agents"),
                                AgentStates,
                            self.cb_pedsim_data )
        else:
            type_detector = str(rospy.get_param("/set_detector","obstacle_detector"))
            
            if type_detector == "obstacle_detector":
                self.sub_peds = rospy.Subscriber(self.ns_prefix("obstacles"),
                                    Obstacles,
                                self.cb_ob_det )
                
            elif type_detector == "dr_spaam":
                self.sub_peds = Subscriber(self.ns_prefix("dr_spaam_detections"),PoseArray) #-> 360 3.5 
                self.sub_peds_vel = Subscriber(self.ns_prefix("dr_spaam_vel"),PoseArray)
                subList = [self.sub_peds, self.sub_peds_vel]
                self.ats = ApproximateTimeSynchronizer(subList, queue_size=1, slop=1)
                self.ats.registerCallback(self.cb_dr_spaam)
                self.listener = tf.TransformListener()
            else:
                self.sub_peds = rospy.Subscriber(self.ns_prefix("pedsim_simulator/simulated_agents"),
                                    AgentStates,
                                self.cb_pedsim_data )

        self.sub_odom  = rospy.Subscriber(self.ns_prefix("odom"),
                            Odometry,
                            self.cb_odom )
        self.ns = self.ns.replace("/",'')
        self.X = 0
        self.Y = 0

    def cb_ob_det(self, msg):
        crowd = msg.circles
        self.cluster = Clusters()

        for i,human in enumerate(crowd):
            tmp_point = Point()
            tmp_vel = Vector3()
            tmp_id = i
            tmp_point.x = human.pose.position.x
            tmp_point.y = human.pose.position.y 
            tmp_vel.x = human.pose.position.x 
            tmp_vel.y = human.pose.position.y
            self.cluster.mean_points.append(tmp_point)
            self.cluster.velocities.append(tmp_point)
            self.cluster.labels.append(tmp_id)
        self.pub_cluster.publish(self.cluster)

    def cb_dr_spaam(self,msg_pos,msg_vel):
        crowd = msg_pos.poses
        vels = msg_vel.poses
        self.cluster = Clusters()
        (trans, rot) = self.listener.lookupTransform(self.ns+'/map', msg_pos.header.frame_id, rospy.Time(0))
        for index in range(len(crowd)):
            tmp_point = Point()
            tmp_vel = Vector3()
            tmp_id = index
            tmp_point.x = crowd[index].position.x + trans[0] # transfrom to map
            tmp_point.y = crowd[index].position.y + trans[1] # transfrom to map
            tmp_vel.x = vels[index].position.x
            tmp_vel.y = vels[index].position.y
            self.cluster.mean_points.append(tmp_point)
            self.cluster.velocities.append(tmp_point)
            self.cluster.labels.append(tmp_id)
        self.pub_cluster.publish(self.cluster)




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
