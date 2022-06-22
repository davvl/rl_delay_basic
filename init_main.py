# SPDX-License-Identifier: Apache-2.0
import gym
from delayed_env import DelayedEnv
import wandb
import warnings

# from keras import backend as K
from tensorflow.python.keras import backend as K
from rand_delayed_env import RandDelayedEnv
from rand_delays import Delay

import tensorflow as tf


config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)


def init_main(agent_type='rand_delayed', env_name='CartPole-v1', delay_value='5', seed='1', total_steps='1000', walking_p='0'):
    hyperparameter_defaults = dict(
        is_delayed_agent=False,
        double_q=True,
        delay_value=int(delay_value),
        epsilon_decay=0.999, # Cartpole: 0.999, Acrobot: 0.9999, MountainCar: 0.99999
        epsilon_min=0.001, #0.001
        learning_rate=0.005, # Cartpole & Acrobot: 0.005, #mountainCar: 0.0001
        seed=int(seed),
        epsilon=1.0,
        use_m_step_reward=False,
        use_latest_reward=False,
        use_reward_shaping=True,
        physical_noise_std_ratio=0.1, # default: 0.1
        env_name=env_name, #'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'
        train_freq=1,
        target_network_update_freq=300,
        use_learned_forward_model=True,
        agent_type=agent_type, #'delayed', 'augmented', 'oblivious', 'rand_delayed'
        action_rand_delay=True,  # True, False
        obs_rand_delay=False,  # True, False
        delay_known=False,  # True, False
        walking_p=float(walking_p),
        total_steps=int(total_steps),
    )
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    wandb.run.name = config.agent_type + ' ' + str(config.delay_value) + ' walking_p=' + str(config.walking_p) +\
                     ' delay_known=' + str(config.delay_known) + ' a=' + str(config.action_rand_delay)
                     #+' o=' + str(config.obs_rand_delay)
    #wandb.run.save()
    if 'CartPole' in config.env_name or 'Acrobot' in config.env_name:
        try:
            orig_env = gym.make(config.env_name, physical_noise_std_ratio=config.physical_noise_std_ratio)
        except TypeError as e:
            warnings.warn('{} gym env has not been modified as needed to support added noise. See README.md for '
                          'instructions.\nRunning original noiseless version instead.'.format(config.env_name))
            orig_env = gym.make(config.env_name)
    else:
        orig_env = gym.make(config.env_name)
    print('config.env_name= {}, orig_env={}'.format(config.env_name, orig_env))

    # orig_env = DiscretizeActions(orig_env) # for mujoco envs
    delay = None
    if config.agent_type == 'rand_delayed':
        delay = Delay(config.delay_value, action=config.action_rand_delay, obs=config.obs_rand_delay,
                      delay_known=config.delay_known, p=config.walking_p)
        delayed_env = RandDelayedEnv(orig_env, config.delay_value, delay)
    else:
        delayed_env = DelayedEnv(orig_env, config.delay_value)

    state_size = orig_env.observation_space.shape#[0]
    if not delayed_env.is_atari_env:
        state_size = state_size[0]
    action_size = orig_env.action_space.n
    done = False
    batch_size = 32
    return config, delayed_env, state_size, action_size, done, batch_size, delay





