from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
import matplotlib.pyplot as plt
import cv2
class DeepMindControl:

    def __init__(self, env):

        self._env = env
        self._camera = 0


    def step(self, action):
        time_step = self._env.step(action)

        obs = dict(time_step.observation)
        obs['image'], obs['depth'] = self.render()
        reward = time_step.reward
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'], obs['depth'] = self.render()
        return obs

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()
    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")

        rgb = self._env.physics.render(height=84,width=84, camera_id=0)
        depth = self._env.physics.render(height=84,width=84, camera_id=0, depth=True)

        rgb=np.mean(rgb,axis=-1)/255
        depth=(depth<50)*depth


        depth/=15
        return rgb, depth

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            #print(time_step)
            #discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='image'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key


    def _transform_observation(self, obs):

        obs[self._pixels_key] = np.stack(list(self._frames), axis=0)

        return obs



    def reset(self):
        obs = self._env.reset()

        for _ in range(self._num_frames):
            self._frames.append(obs[self._pixels_key])
        return self._transform_observation(obs)

    def step(self, action):

        obs, reward, done, info = self._env.step(action)
        self._frames.append(obs[self._pixels_key])
        return self._transform_observation(obs),reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class Mask:

    def __init__(self, env):
        self._env = env

        self.mask=np.ones((84,84))
        self.mask1 = self.get_mask(22, 22)
        self.mask2 = self.get_mask(62, 62)

        self.mask3 = self.get_mask(22, 62)
        self.mask4 = self.get_mask(62, 22)
    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._missing_obs(obs)
        self._t += 1
        return obs, reward, done, info
    def get_mask(self,x,y):
        mask=np.ones((84,84))
        mask_center_x=x
        mask_center_y=y
        mask_size=22
        limit_x_up = mask_center_x+mask_size
        limit_x_down = mask_center_x - mask_size
        limit_y_up = mask_center_y + mask_size
        limit_y_down = mask_center_y - mask_size
        if(limit_x_up>84):
            limit_x_up=84
        if (limit_y_up > 84):
            limit_y_up = 84
        if (limit_x_down <0):
            limit_x_down = 0
        if (limit_y_down <0):
            limit_y_down = 0
        for i in range(limit_x_down,limit_x_up):
            for j in range(limit_y_down,limit_y_up):
                mask[i][j]=0
        return mask

    def _missing_obs(self, obs):


        obs['image']=obs['image']*self.mask1
        obs['image'] = obs['image'] * self.mask2
        obs['depth'] = obs['depth'] * self.mask3
        obs['depth'] = obs['depth'] * self.mask4

        return obs

    def reset(self):
        self._t = 0
        self._drop_end = dict()
        obs = self._env.reset()
        obs = self._missing_obs(obs)
        return obs


class Maskdepth:

    def __init__(self, env):
        self._env = env
        self.miss_ratio={ "depth": 1}
        self._t = 0
        self._drop_end = dict()
        self.max_miss_len=15
        self.mask=np.ones((84,84))
        self.mask = self.get_mask()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._missing_obs(obs)
        self._t += 1
        return obs, reward, done, info
    def get_mask(self):
        mask=np.ones((84,84))
        mask_center_x=20
        mask_center_y=20
        mask_size=20
        limit_x_up = mask_center_x+mask_size
        limit_x_down = mask_center_x - mask_size
        limit_y_up = mask_center_y + mask_size
        limit_y_down = mask_center_y - mask_size
        if(limit_x_up>84):
            limit_x_up=84
        if (limit_y_up > 84):
            limit_y_up = 84
        if (limit_x_down <0):
            limit_x_down = 0
        if (limit_y_down <0):
            limit_y_down = 0
        for i in range(limit_x_down,limit_x_up):
            for j in range(limit_y_down,limit_y_up):
                mask[i][j]=0
        return mask

    def _missing_obs(self, obs):
        obs_c = obs.copy()

        for key, value in obs.items():
            if key in self.miss_ratio:
                value_f = value
                flag = np.array([1]).astype(np.float16)
                if self._t < self._drop_end.get(key, 0):
                    value_f =  value*self.mask
                    flag = 0 * flag
                elif np.random.rand() <= self.miss_ratio.get(key, -1.0):

                    value_f = value*self.mask
                    flag = 0 * flag
                    self._drop_end[key] = self._t + np.random.randint(1, self.max_miss_len)
                obs_c[key] = value_f
                obs_c[key + '_flag'] = flag
        return obs_c

    def reset(self):
        self._t = 0
        self._drop_end = dict()
        obs = self._env.reset()
        obs = self._missing_obs(obs)
        return obs


class Maskimage:

    def __init__(self, env):
        self._env = env
        self.miss_ratio = {"image": 1}
        self._t = 0
        self._drop_end = dict()
        self.max_miss_len = 15
        self.mask = np.ones((84, 84))
        self.mask = self.get_mask()
    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._missing_obs(obs)
        self._t += 1
        return obs, reward, done, info

    def get_mask(self):
        mask = np.ones((84, 84))
        mask_center_x = np.random.randint(20, 64)
        mask_center_y = np.random.randint(20, 64)
        mask_size = 10
        limit_x_up = mask_center_x + mask_size
        limit_x_down = mask_center_x - mask_size
        limit_y_up = mask_center_y + mask_size
        limit_y_down = mask_center_y - mask_size
        if (limit_x_up > 84):
            limit_x_up = 84
        if (limit_y_up > 84):
            limit_y_up = 84
        if (limit_x_down < 0):
            limit_x_down = 0
        if (limit_y_down < 0):
            limit_y_down = 0
        for i in range(limit_x_down, limit_x_up):
            for j in range(limit_y_down, limit_y_up):
                mask[i][j] = 0
        return mask

    def _missing_obs(self, obs):
        obs_c = obs.copy()

        for key, value in obs.items():
            if key in self.miss_ratio:
                value_f = value
                flag = np.array([1]).astype(np.float16)
                if self._t < self._drop_end.get(key, 0):
                    value_f = value * self.mask
                    flag = 0 * flag
                elif np.random.rand() <= self.miss_ratio.get(key, -1.0):

                    value_f = value * self.mask
                    flag = 0 * flag
                    self._drop_end[key] = self._t + np.random.randint(1, self.max_miss_len)
                obs_c[key] = value_f
                obs_c[key + '_flag'] = flag
        return obs_c

    def reset(self):
        self._t = 0
        self._drop_end = dict()
        obs = self._env.reset()
        obs = self._missing_obs(obs)
        return obs

class MissingMultimodal:

    def __init__(self, env):
        self._env = env
        self.miss_ratio={"image": 1, "depth": 1}
        self._t = 0
        self._drop_end = dict()
        self.max_miss_len=15
    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._missing_obs(obs)
        self._t += 1
        return obs, reward, done, info

    def _missing_obs(self, obs):
        obs_c = obs.copy()

        for key, value in obs.items():
            if key in self.miss_ratio:
                value_f = value
                flag = np.array([1]).astype(np.float16)
                if self._t < self._drop_end.get(key, 0):

                    value_f =  value+np.random.random(size=value.shape)*0.5
                    flag = 0 * flag
                elif np.random.rand() <= self.miss_ratio.get(key, -1.0):

                    value_f = value+np.random.random(size=value.shape)*0.5
                    flag = 0 * flag
                    self._drop_end[key] = self._t + np.random.randint(1, self.max_miss_len)
                obs_c[key] = value_f
                obs_c[key + '_flag'] = flag
        return obs_c

    def reset(self):
        self._t = 0
        self._drop_end = dict()
        obs = self._env.reset()
        obs = self._missing_obs(obs)
        return obs

class multimodal_wrapper():
    def __init__(self, env, mode ):
        super(multimodal_wrapper, self).__init__()

        self.mode = mode
        self.env = env


        # 初始化机器人传感器信息

    def get_obs(self, obs):
        state = []

        if(self.mode=='mm'):

            state.append(obs['image'])
            state.append(obs['depth'])
        elif (self.mode == 'depth'):
            return obs['depth']
        else:
            return obs['image']

        return state

    def reset(self):
        obs = self.env.reset()
        state = self.get_obs(obs)
        return state

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state = self.get_obs(obs)
        return state, reward, done, info


def make(env,  action_repeat,mode='rgb'):

    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    env = DeepMindControl(env)
    env = Mask(env)

     #env=MissingMultimodal(env)
    if(mode=='image' or mode=='mm'):
        env = FrameStackWrapper(env, 3,'image')
    if (mode == 'depth' or mode == 'mm'):
        env = FrameStackWrapper(env, 3, 'depth')
    env = multimodal_wrapper(env,mode)

    # add renderings for clasical tasks

    # stack several frames

    #env = ExtendedTimeStepWrapper(env)
    return env


if __name__ == '__main__':
    env = suite.load('walker', 'walk',task_kwargs={'random': 0},visualize_reward=False)
    env = make(env,action_repeat=2,mode='mm')
    plt.ion()
    env.reset()
    for i in range(50000):
        plt.cla()
        obs,reward,done,info=env.step([0.2])


        if(done):
            break


        plt.imshow(np.concatenate([obs[0][0],obs[1][0]]))
        np.save('obs0.npy',obs[0][0])
        np.save('obs1.npy',obs[1][0])
        
        

        plt.pause(0.1)
