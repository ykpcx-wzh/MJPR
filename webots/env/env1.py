import matplotlib.pyplot as plt

from controller import Robot, Camera, Field
from controller import LidarPoint, Motor
from controller import Lidar, DistanceSensor, PositionSensor, Supervisor, GPS
import os
import numpy as np
import math
import random

def is_place(x,y,object1,object2):



    if (x > 0.4 or x < -0.4 or y < -0.4 or y > 0.4):
        return False

    for ob in object1:
        r=pow((x-ob[0])**2+(y-ob[1])**2,0.5)
        if(r<0.15):
            return False

    for ob in object2:
        r = pow((x - ob[0]) ** 2 + (y - ob[1]) ** 2, 0.5)

        if (r < 0.15):
            return False

    return True

def wall_detection(x,y):
    if(x>0.42 or x<-0.42 or y>0.42 or y<-0.42):
        return True
    else:
        return False


def box_detection(x,y,object):
    for ob in object:
        r = pow((x - ob[0]) ** 2 + (y - ob[1]) ** 2, 0.5)
        if (r < 0.115):
            return True

    else:
        return False

def ball_detection(x,y,object):
    for ob in object:
        r = pow((x - ob[0]) ** 2 + (y - ob[1]) ** 2, 0.5)
        if (r < 0.065):
            return True
        print('ball', r)
    else:
        return False



class env():
    def __init__(self, seed, Max_step):
        super(env, self).__init__()

        self.robot = Supervisor()

        self.name = self.robot.getFromDef("aqua")
        # 初始化机器人传感器信息
        self.trans = self.name.getField("translation")
        self.rotation = self.name.getField("rotation")
        self.timestep = int(self.robot.getBasicTimeStep())
        self.camera = self.robot.getDevice('camera')
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.camera.enable(self.timestep)
        self.left_wheel_motor = self.robot.getDevice('left wheel motor')
        self.right_wheel_motor = self.robot.getDevice('right wheel motor')



        self.left_wheel_motor.setPosition(float('inf'))
        self.right_wheel_motor.setPosition(float('inf'))

        self.object1=[]
        self.object2 = []
        self.box=[]
        self.box.append(self.robot.getFromDef("box1"))
        self.box.append(self.robot.getFromDef("box2"))
        self.box.append(self.robot.getFromDef("box3"))
        self.box.append(self.robot.getFromDef("box4"))
        self.ball=[]
        self.ball.append(self.robot.getFromDef("target1"))
        self.ball.append(self.robot.getFromDef("target2"))
        self.ball.append(self.robot.getFromDef("target3"))

        self.ball_num = 3
        #print(self.box[0])



        # 目标位置

        # 设定仿真加速
        self.robot.simulationSetMode(2)

        # 设立随机种子
        self.seed = seed
        self.set_seed = self.robot.getFromDef('n')
        self.set_seed.getField("randomSeed").setSFInt32(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        # 设置速度上限以及仿真时间
        self.speed = self.left_wheel_motor.max_velocity
        self.delay_time = 20
        self.max_step = Max_step
        self.now_step = 0
        self.x=0
        self.y=0
        # 记录距离


    def get_state(self, action=np.zeros(2)):
        img = self.camera.getImageArray()
        img = np.array(img)
        #print(img.shape)
        #print(img.shape)


        #img = cv2.resize(img, (80, 80))



        # np.save('picture.npy', img)

        lidar_data=self.lidar.getRangeImage()
        for i in range(len(lidar_data)):
            if(math.isinf(lidar_data[i])):
                lidar_data[i] = 0
            if (math.isnan(lidar_data[i])):
                lidar_data[i] = 0

        pos = np.array(self.trans.getSFVec3f())
        rot = np.array(self.rotation.getSFRotation())
        pos_rot = np.zeros(5)
        pos_rot[0] = pos[0]
        pos_rot[1] = pos[1]
        a, b, pos_rot[2] = self.quaternion_to_euler(rot[0], rot[1], rot[2], rot[3])
        pos_rot[3] = (pos[0]-self.x)*50
        pos_rot[4] = (pos[1]-self.y)*50
        self.x=pos[0]
        self.y=pos[1]
        #print(pos_rot)
        #state = [ np.average(img.reshape((84,84,3)),axis=-1).reshape((1,84,84))/255, np.array(lidar_data).reshape((1,360)), pos_rot.reshape((1,5))]
        state = [np.average(img.reshape((84,84,3)),axis=-1).reshape((1,84,84))/255,
                 np.array(lidar_data).reshape((1, 360)), pos_rot.reshape((1, 5))]
        '''
        state = [np.average(img.reshape((84,84,3)),axis=-1).reshape((1,84,84))/255,
                 np.array(lidar_data).reshape((1, 360)), pos_rot.reshape((1, 5))]
                 '''
        return state

    def quaternion_to_euler(self, qx, qy, qz, qw, degree_mode=0):

        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))

        pitch = 0
        sinp = 2 * (qw * qy - qz * qx)
        if (abs(sinp) >= 1):
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        # degree_mode=1:【输出】是角度制，否则弧度制
        if degree_mode == 1:
            roll = np.rad2deg(roll)
            pitch = np.rad2deg(pitch)
            yaw = np.rad2deg(yaw)
        euler = np.array([roll, pitch, yaw])

        return euler

    # 欧拉角转四元数
    # ================OKOK
    def euler_to_quaternion(self, euler, degree_mode=1):
        roll, pitch, yaw = euler
        # degree_mode=1:【输入】是角度制，否则弧度制
        if degree_mode == 1:
            roll = np.deg2rad(roll)
            pitch = np.deg2rad(pitch)
            yaw = np.deg2rad(yaw)

        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        q = np.array([qw, qx, qy, qz])
        return q

    def reset(self):

        # 重新设置初始信息
        self.robot.simulationReset()
        self.robot.step(self.timestep)
        self.left_wheel_motor.setPosition(float('inf'))
        self.right_wheel_motor.setPosition(float('inf'))

        self.now_step = 0
        # 设立小车初始位置
        self.set_car_target()

        self.robot.step(self.timestep)
        return self.get_state()

    def set_car_target(self):
        #放置四个盒子
        self.object1=[]
        self.object2=[]
        for i in range(4):
            x = random.random() * 0.8 - 0.4
            y = random.random() * 0.8 - 0.4
            while(is_place(x,y,object1=self.object1,object2=self.object2)==False):
                x = random.random() * 0.8 - 0.4
                y = random.random() * 0.8 - 0.4
            self.object1.append([x,y])
            wz=self.box[i].getField('translation')
            wz.setSFVec3f([x,y,0.05])
        #放置小球
        for i in range(self.ball_num):
            x = random.random() * 0.8 - 0.4
            y = random.random() * 0.8 - 0.4
            while(is_place(x,y,object1=self.object1,object2=self.object2)==False):
                x = random.random() * 0.8 - 0.4
                y = random.random() * 0.8 - 0.4
            self.object2.append([x,y])
            wz=self.ball[i].getField('translation')
            wz.setSFVec3f([x,y,0.04])


        # 放置小车
        x = random.random() * 0.7 - 0.35
        y = random.random() * 0.7 - 0.35
        while (is_place(x, y, object1=self.object1,object2=self.object2) == False):
            x = random.random() * 0.7 - 0.35
            y = random.random() * 0.7 - 0.35
        self.x=x
        self.y=y
        self.trans.setSFVec3f([x, y, 0])

        #print(self.ball)




        '''
        x = 1.2
        y = 1.1
        self.target[0] = -1
        self.target[1] = -1
        self.trans.setSFVec3f([x, y, 0])
        r = (x +1) ** 2 + (y+1) ** 2
        self.last_distance = pow(r, 0.5)
        '''

    def delay(self, n):
        for i in range(n):
            for j in range(self.ball_num):
                self.ball[j].setVelocity([0,0])
            self.robot.step(self.timestep)

    def step(self, action):
        # 给定动作

        self.now_step += 1

        left_speed = (action[0] + 1) * self.speed/2
        right_speed = (action[1] + 1) * self.speed/2
        self.left_wheel_motor.setVelocity(left_speed)
        self.right_wheel_motor.setVelocity(right_speed)


        self.delay(self.delay_time)

        state = self.get_state(action)
        done = False
        target = False
        reward = 0
        pos = state[2][0]

        # 检测收集到球，并给予正奖励
        for i in range(3):


            this_distance = math.pow(((pos[0] - self.object2[i][0]) ** 2 + (pos[1] - self.object2[i][1]) ** 2), 0.5)

            if (this_distance < 0.1):

                reward += 10

                #重新放置小球
                x = random.random() * 0.8 - 0.4
                y = random.random() * 0.8 - 0.4
                while (is_place(x, y, object1=self.object1, object2=self.object2) == False):
                    x = random.random() * 0.8 - 0.4
                    y = random.random() * 0.8 - 0.4
                self.object2[i]=[x, y]

                wz = self.ball[i].getField('translation')
                wz.setSFVec3f([x, y, 0.15])



        # 触碰到箱子和墙壁给予负奖励
        if (wall_detection(pos[0], pos[1]) == True):
            reward -= 10.0
            target = True

        elif (box_detection(pos[0], pos[1], self.object1) == True):
            reward -= 10.0
            target = True

        reward += action[0] * action[1] * 0.1 - 0.1
        # reward-=abs(action[0]-action[1])*0.01-0.01

        if (self.now_step >= self.max_step):
            done = True
        # reward3=-abs(action[0]-action[1])/1000

        # print(box_detection(pos[0],pos[1],self.object1))
        # print(ball_detection(pos[0], pos[1], self.object2))
        #state[2] = np.zeros((1,5))
        return state, reward, done, target


if __name__ == '__main__':
    env = env(1, 100)
    plt.ion()
    for e in range(100):
        rw=0
        env.reset()
        print(e)

        for i in range(10000):
            action1= 0.5
            action2 = 0
            state, reward, done,target = env.step([action1, action2])

            plt.imshow(state[0].reshape((84,84)))
            plt.pause(0.0001)
            plt.clf()
            print(i,reward)
            #path=r'D:/Python_file/'
            #np.save(path+'picture2.npy', state[1])
            if(target or done):
                break
            rw+=reward

            #print(reward,target)
        print(rw)
