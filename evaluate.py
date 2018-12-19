import sys
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from agent.agent import Agent
from environment import SimpleTradeEnv
from functions import formatPrice

if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]

# Load agent
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1] - 1
agent = Agent(window_size, True, model_name)

# Environment
env = SimpleTradeEnv(stock_name, window_size, agent)

# Initialization before starting an episode
state = env.reset()
agent.inventory = []
done = False

# Loop in an episode
t = 0
ts_buy = []
ts_sell = []

while not done:
	could_buy = len(agent.inventory) > 0
	action = agent.act(state)
	next_state, _, done, _ = env.step(action)
	state = next_state

	if action == 1:
		ts_buy.append(t)
	elif action == 2 and could_buy:
		ts_sell.append(t)

	t += 1

	if done:
		print("--------------------------------")
		print("Total Profit: " + formatPrice(env.total_profit))
		print("--------------------------------")

plt.figure()
data = np.array(env.data)
ts = np.arange(len(env.data)).astype(int)
ts_buy = np.array(ts_buy).astype(int)
ts_sell = np.array(ts_sell).astype(int)
plt.plot(ts, data[ts])
plt.scatter(ts_buy, data[ts_buy], c="r", label="Buy")
plt.scatter(ts_sell, data[ts_sell], c="b", label="Sell")
plt.legend()
plt.show()
