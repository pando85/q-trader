import sys
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
from agent.agent import Agent
from environment import SimpleTradeEnv
from functions import formatPrice

if len(sys.argv) != 4:
	print("Usage: python evaluate.py [config] [stock] [model_name]")
	exit()

config_file = sys.argv[1]
stock_name = sys.argv[2]
model_name = sys.argv[3]

with open(config_file) as f:
	yaml = YAML()
	config = yaml.load(f)

# Load agent
window_size = config["window_size"]
agent = Agent(window_size, True, model_name,
							result_dir=config["result_dir"])

# Environment
env = SimpleTradeEnv(stock_name, window_size, agent,
										 inventory_max=config["inventory_max"])

# Initialization before starting an episode
state = env.reset()
agent.inventory = []
done = False

# Loop in an episode
t = 0
ts_buy = []
ts_sell = []

while not done:
	could_buy = len(agent.inventory) < config["inventory_max"]
	could_sell = len(agent.inventory) > 0
	action = agent.act(state)
	next_state, _, done, _ = env.step(action)
	state = next_state

	if action == 1 and could_buy:
		ts_buy.append(t)
	elif action == 2 and could_sell:
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
