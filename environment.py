import numpy as np
from functions import getStockDataVec, formatPrice, sigmoid


# returns an an n-day state representation ending at time t
def getState(data, t, n, agent):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return agent.modify_state(np.array([res]))


class SimpleTradeEnv(object):
  def __init__(self, stock_name, window_size, agent):
    self.data = getStockDataVec(stock_name)
    self.window_size = window_size
    self.agent = agent

  def step(self, action):
    # 0: Sit
    # 1: But
    # 2: Sell
    assert(action in (0, 1, 2))

    # State transition
    next_state = getState(self.data, self.t + 1, self.window_size + 1, self.agent)

    # Reward
    if action == 0:
      reward = 0

    elif action == 1:
      reward = 0
      self.agent.inventory.append(self.data[self.t])
      self.total_profit -= self.data[self.t]
      print("Buy: " + formatPrice(self.data[self.t]))

    else:
      if len(self.agent.inventory) > 0:
        bought_price = self.agent.inventory.pop(0)
        reward = self.data[self.t] - bought_price
        # reward = max(self.data[self.t] - bought_price, 0)
        self.total_profit += self.data[self.t]
        print("Sell: " + formatPrice(self.data[self.t]) +
              " | Profit: " + formatPrice(reward))
      else:
        reward = 0 # try to sell, but con't do

    done = True if self.t == len(self.data) - 2 else False
    self.t += 1

    return next_state, reward, done, {}

  def reset(self):
    self.t = 0
    self.total_profit = 0
    return getState(self.data, self.t, self.window_size + 1, self.agent)
