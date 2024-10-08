
import copy
import json
import math
import os
import random
import sys
import time

from env import env1
from drq_aug import drq_aug

import hydra
import numpy as np
import torch
from logger import Logger
from omegaconf import DictConfig
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from mjpr import mjpr


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self,
                 obs_shape1,
                 obs_shape2,
                 obs_shape3,
                 action_shape,
                 capacity,
                 batch_size,
                 device,
                 image_size=84,
                 transform=None,
                 auxiliary_task_batch_size=64,
                 jumps=5):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.auxiliary_task_batch_size = auxiliary_task_batch_size
        self.jumps = jumps
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32

        self.image_obses = np.empty((capacity, *obs_shape1), dtype=obs_dtype)
        self.laser_obses = np.empty((capacity, *obs_shape2), dtype=obs_dtype)
        self.pos_obses = np.empty((capacity, *obs_shape3), dtype=obs_dtype)
        self.next_image_obses = np.empty((capacity, *obs_shape1), dtype=obs_dtype)
        self.next_laser_obses = np.empty((capacity, *obs_shape2), dtype=obs_dtype)
        self.next_pos_obses = np.empty((capacity, *obs_shape3), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)  # for infinite bootstrap
        self.real_dones = np.empty((capacity, 1), dtype=np.float32)  # for auxiliary task

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.current_auxiliary_batch_size = batch_size

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.image_obses[self.idx], obs[0])
        np.copyto(self.laser_obses[self.idx], obs[1])
        np.copyto(self.pos_obses[self.idx], obs[2])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_image_obses[self.idx], next_obs[0])
        np.copyto(self.next_laser_obses[self.idx], next_obs[1])
        np.copyto(self.next_pos_obses[self.idx], next_obs[2])
        np.copyto(self.not_dones[self.idx], not done)  # "not done" is always True
        np.copyto(self.real_dones[self.idx], done)  # "not done" is always True
        # print(not done, isinstance(done, int))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)
        image_obses = self.image_obses[idxs]
        laser_obses = self.laser_obses[idxs]
        pos_obses = self.pos_obses[idxs]
        next_image_obses = self.next_image_obses[idxs]
        next_laser_obses = self.next_laser_obses[idxs]
        next_pos_obses = self.next_pos_obses[idxs]

        image_obses = torch.as_tensor(image_obses, device=self.device).float()
        laser_obses = torch.as_tensor(laser_obses, device=self.device).float()
        pos_obses = torch.as_tensor(pos_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_image_obses = torch.as_tensor(next_image_obses, device=self.device).float()
        next_laser_obses = torch.as_tensor(next_laser_obses, device=self.device).float()
        next_pos_obses = torch.as_tensor(next_pos_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return [image_obses,laser_obses,pos_obses], actions, rewards, [next_image_obses,next_laser_obses,next_pos_obses], not_dones

    # v2
    def sample_spr(self):  # sample batch for auxiliary task
        idxs = np.random.randint(0,
                                 self.capacity - self.jumps -
                                 1 if self.full else self.idx - self.jumps - 1,
                                 size=self.auxiliary_task_batch_size * 2)
        #  size=self.auxiliary_task_batch_size)

        idxs = idxs.reshape(-1, 1)
        step = np.arange(self.jumps + 1).reshape(1, -1)  # this is a range
        idxs = idxs + step

        real_dones = torch.as_tensor(self.real_dones[idxs], device=self.device)  # (B, jumps+1, 1)

        # we add this to avoid sampling the episode boundaries
        valid_idxs = torch.where((real_dones.mean(1) == 0).squeeze(-1))[0].cpu().numpy()

        idxs = idxs[valid_idxs]  # (B, jumps+1)

        idxs = idxs[:self.auxiliary_task_batch_size] if idxs.shape[0] >= self.auxiliary_task_batch_size else idxs
        self.current_auxiliary_batch_size = idxs.shape[0]

        image_obses = torch.as_tensor(self.image_obses[idxs], device=self.device).float()
        laser_obses = torch.as_tensor(self.laser_obses[idxs], device=self.device).float()
        pos_obses = torch.as_tensor(self.pos_obses[idxs], device=self.device).float()

        #next_image_obses = torch.as_tensor(self.next_image_obses[idxs], device=self.device).float()
        #next_depth_obses = torch.as_tensor(self.next_depth_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        #not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)  # (B, jumps+1, 1)

        spr_samples = {
            'image': image_obses.transpose(0, 1).unsqueeze(3),
            'laser': laser_obses.transpose(0, 1).unsqueeze(3),
            'pos': pos_obses.transpose(0, 1),
            'action': actions.transpose(0, 1),
            'reward': rewards.transpose(0, 1),
        }
        return (*self.sample(), spr_samples)

    def sample(self):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        image_obses = torch.as_tensor(self.image_obses[idxs], device=self.device).float()
        laser_obses = torch.as_tensor(self.laser_obses[idxs], device=self.device).float()
        pos_obses = torch.as_tensor(self.pos_obses[idxs], device=self.device).float()
        next_image_obses = torch.as_tensor(self.next_image_obses[idxs], device=self.device).float()
        next_laser_obses = torch.as_tensor(self.next_laser_obses[idxs], device=self.device).float()
        next_pos_obses = torch.as_tensor(self.next_pos_obses[idxs], device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)

        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return [image_obses,laser_obses,pos_obses], actions, rewards,[next_image_obses,next_laser_obses,next_pos_obses], not_dones



    def add_agent(self, agent):
        if hasattr(agent, 'SPR'):
            self.SPR = agent.SPR




class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False



def evaluate(env, agent, num_episodes, L, step):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()

            done = False
            target = False
            episode_reward = 0
            while not (done or target):
                # center crop image

                with eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, target = env.step(action)

                episode_reward += reward


            #L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        #L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        print('eval:',mean_ep_reward,best_ep_reward)
        return mean_ep_reward
        #L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        #L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    return run_eval_loop(sample_stochastically=False)
     


def make_agent(obs_shape, action_shape, args, device):


    if args.agent == 'mjpr':
        return sac(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            transition_model_type=args.transition_model_type,
            transition_model_layer_width=args.transition_model_layer_width,
            jumps=args.jumps,
            latent_dim=args.latent_dim,
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
            model_spr_weight=args.fp_loss_weight,
            auxiliary_task_lr=args.auxiliary_task_lr,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim)   

    else:
        assert 'agent is not supported: %s' % args.agent


@hydra.main(config_name="config")
def main(args: DictConfig) -> None:
    
    reward_all=[]
    args.init_steps *= args.action_repeat
    args.log_interval *= args.action_repeat
    args.actor_update_freq *= args.action_repeat
    args.critic_target_update_freq *= args.action_repeat
    args.seed = args.seed or args.seed_and_gpuid[0]
    args.gpuid = args.gpuid or args.seed_and_gpuid[1]
    args.domain_name = args.domain_name or args.env_name.split('/')[0]

    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
    torch.cuda.set_device(args.gpuid)
    set_seed_everywhere(args.seed)
    seed=int(args.seed)
    env = env1.env(seed,args.task_step)



    # stack several consecutive frames together


    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m%dT%H", ts)
    # exp_name = 'aug_then_expand_AN10'    # re-run
    exp_name = 'ETA-w2'    # re-run
    env_name = 'env1'
    exp_name = env_name + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed) + '-' + args.agent

    if args.agent == 'spr_sac':
        exp_name = exp_name + '-k' + str(args.jumps)
    if args.agent == 'cycdm_sac':
        exp_name = exp_name + '-' + args.cycle_mode
        # exp_name = exp_name.replace('fp', f'fp{args.jumps}')
        # exp_name = exp_name.replace('cycle', f'cycle{args.cycle_steps}')


    args.work_dir = args.work_dir + '/' + f'{args.agent}_dmc' + '/' + exp_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = [int(args.action_shape)]

    if args.encoder_type == 'pixel':
        obs_shape = (args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (args.frame_stack,
                             args.pre_transform_image_size,
                             args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = ReplayBuffer(
        obs_shape1=pre_aug_obs_shape,
        obs_shape2=args.lidar_shape,
        obs_shape3=args.pos_dim,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        auxiliary_task_batch_size=args.auxiliary_task_batch_size,
        jumps=args.jumps,
    )

    agent = make_agent(obs_shape=obs_shape,
                       action_shape=action_shape,
                       args=args,
                       device=device)
    replay_buffer.add_agent(agent)

    L = Logger(args.work_dir, use_tb=args.save_tb, use_wandb=args.wandb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    frame = [np.zeros((84, 84)), np.zeros((84, 84)), np.zeros((84, 84))]
    eval_num=0
    eval_num += 1
    # L.log('eval/episode', episode, step)

    r = evaluate(env, agent, args.num_eval_episodes, L, 0)
    reward_all.append(r)
    np.save('reward_' + 'env1' + '_' + str(args.agent) + str(args.seed) + '.npy', np.array(reward_all))
    for step in range(0, args.num_env_steps, args.action_repeat):
        # evaluate agent periodically





        if (done or target):
            if step / args.eval_freq >= eval_num:
                eval_num += 1
                # L.log('eval/episode', episode, step)

                r = evaluate(env, agent, args.num_eval_episodes, L, step)
                reward_all.append(r)
                np.save('reward_' + 'env1' + '_' + str(args.agent) + str(args.seed) + '.npy', np.array(reward_all))

            if step > 0:
                '''
                if step % args.log_interval == 0:
                    #L.log('train/duration', time.time() - start_time, step)
                    #L.dump(step)
                    print('duration', time.time() - start_time, step)
                start_time = time.time()
                '''
                #L.log('train/episode_reward', episode_reward, step)
            agent.save('env1',args.agent,str(args.seed))
            agent.save_spr('env1',args.agent,str(args.seed))
            print('episode_reward', episode_reward, step)
            obs = env.reset()
            done = False
            target = False
            episode_reward = 0

            episode += 1




        # sample action for data collection
        if step < args.init_steps:
            action = np.random.rand(args.action_shape)*2-1
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, target = env.step(action)

        # allow infinit bootstrap


        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, target)
        obs=next_obs




if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
