import gym;
import gym_np_snake

from baselines import logger
from baselines.acktr.acktr_disc import learn
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2.policies import CnnPolicy
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import os

import numpy as np
from baselines.common.cmd_util import make_atari_env, atari_arg_parser


snake_env = gym.make('SnakeNp-v0')

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    


def train(num_timesteps, seed, num_cpu):
    env = VecFrameStack(make_custom_env('SnakeNp-v0',1,3), 1)
    policy_fn = CnnPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu)
    env.close()


if __name__ == "__main__":
    logger.configure()
    train(num_timesteps=100, seed=10, num_cpu=4)
