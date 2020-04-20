import itertools

import gym
import numpy as np
from gym import spaces

from sklearn.preprocessing import StandardScaler


class MultiStockEnvUwrapped:
  """ 
  Credits to: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/tf2.0/rl_trader.py 
  """

  def __init__(self, data, initial_investment):
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape
    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None
    self.action_space = np.arange(3**self.n_stock)
    # action permutations
    self.action_list = list(
        map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
    # calculate size of state
    self.state_dim = self.n_stock * 2 + 1
    # reset environment
    self.reset()

  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment
    return self._get_obs()

  def step(self, action):
    assert action in self.action_space
    # get current value before performing the action
    prev_val = self._get_val()
    # update price, i.e. go to the next day
    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]
    # perform the trade
    self._trade(action)
    # get the new value after taking the action
    cur_val = self._get_val()
    # reward is the increase in porfolio value
    reward = cur_val - prev_val
    # done if we have run out of data
    done = self.cur_step == self.n_step - 1
    # store the current value of the portfolio here
    info = {'cur_val': cur_val}
    # conform to the Gym API
    return self._get_obs(), reward, done, info

  def _get_obs(self):
    obs = np.empty(self.state_dim)
    obs[:self.n_stock] = self.stock_owned
    obs[self.n_stock:2*self.n_stock] = self.stock_price
    obs[-1] = self.cash_in_hand
    return obs

  def _get_val(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

  def _trade(self, action):
    action_vec = self.action_list[action]
    # determine which stocks to buy or sell
    sell_index = []  # stores index of stocks we want to sell
    buy_index = []  # stores index of stocks we want to buy
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)
    # sell any stocks we want to sell
    # then buy any stocks we want to buy
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
    if buy_index:
      # NOTE: when buying, we will loop through each stock we want to buy,
      #       and buy one share at a time until we run out of cash
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1  # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False


class MultiStockEnv(gym.Env):
  """Environment that follows gym interface, with obs. space standardization"""

  def __init__(self, data, initial_investment):
    super(MultiStockEnv, self).__init__()
    self.env = MultiStockEnvUwrapped(data, initial_investment)
    self.action_space = spaces.Discrete(self.env.action_space.size)
    self.observation_space = spaces.Box(
        low=-4, high=4, shape=(self.env.state_dim,), dtype=np.float32)
    self.obs_scaler = self._get_scaler()
    self.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    return self.obs_scaler.transform([obs])[0], reward, done, info

  def reset(self):
    obs = self.env.reset()
    return self.obs_scaler.transform([obs])[0]

  def render(self):
    pass

  def close(self):
    pass

  def _get_scaler(self):
    states = []
    self.env.reset()
    for _ in range(self.env.n_step):
      action = np.random.choice(self.env.action_space)
      obs, _, done, _ = self.env.step(action)
      states.append(obs)
      if done:
        break
    self.env.reset()
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler
