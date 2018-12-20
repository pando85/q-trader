import numpy as np
from functions import getStockDataVec, formatPrice, sigmoid


# returns an an n-day state representation ending at time t
def getState(data, t, n, agent):
  d = t - n + 1
  block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad # with t0
  res = []
  for i in range(n - 1):
    res.append(sigmoid(block[i + 1] - block[i]))

  return agent.modify_state(np.array([res]))


class SimpleTradeEnv(object):
  def __init__(self, stock_name, window_size, agent, inventory_max,
               clip_reward=True, reward_for_buy=-20, print_trade=True):
    self.data = getStockDataVec(stock_name)
    self.window_size = window_size
    self.agent = agent
    self.print_trade = print_trade
    self.reward_for_buy = reward_for_buy
    self.clip_reward = clip_reward
    self.inventory_max = inventory_max

  def step(self, action):
    # 0: Sit
    # 1: Buy
    # 2: Sell
    assert(action in (0, 1, 2))

    # Reward
    if action == 0:
      reward = 0

    elif action == 1:
      if len(self.agent.inventory) < self.inventory_max:
        reward = self.reward_for_buy
        self.agent.inventory.append(self.data[self.t])

        if self.print_trade:
          print("Buy: " + formatPrice(self.data[self.t]))
      else:
        reward = 0
        if self.print_trade:
          print("Buy: not possible")

    else:
      if len(self.agent.inventory) > 0:
        bought_price = self.agent.inventory.pop(0)
        profit = self.data[self.t] - bought_price
        reward = max(profit, 0) if self.clip_reward else profit
        self.total_profit += profit

        if self.print_trade:
          print("Sell: " + formatPrice(self.data[self.t]) +
                " | Profit: " + formatPrice(reward))
      else:
        if self.print_trade:
          print("Sell: not possible")
        reward = -10 # try to sell, but con't do, penalty

    # State transition
    next_state = getState(self.data, self.t + 1, self.window_size + 1,
                          self.agent)

    done = True if self.t == len(self.data) - 2 else False
    self.t += 1

    return next_state, reward, done, {}

  def reset(self):
    self.t = 0
    self.total_profit = 0
    return getState(self.data, self.t, self.window_size + 1, self.agent)
