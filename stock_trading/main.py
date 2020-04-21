import argparse
import csv
from urllib.request import urlopen

import pandas as pd
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env

from env import MultiStockEnv

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='./logs', type=str)
    parser.add_argument('--timesteps', default=100000, type=int)
    args = parser.parse_args()

    # Get stocks
    STOCKS_URL = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/aapl_msi_sbux.csv'
    response = urlopen(STOCKS_URL)
    data = pd.read_csv(response).values

    # Init and check the environment
    env = MultiStockEnv(data, initial_investment=20000)
    check_env(env)

    # Train
    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=args.logdir,
                # hyperparameters
                gamma=0.95,
                buffer_size=500,
                exploration_final_eps=0.01,
                double_q=False,
                learning_starts=500,
                target_network_update_freq=30)
    model.learn(total_timesteps=args.timesteps, tb_log_name='stock_trading')
