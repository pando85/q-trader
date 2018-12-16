from agent.agent import Agent
from functions import *
import sys
from environment import SimpleTradeEnv

print(sys.argv)

if len(sys.argv) != 4:
  print("Usage: python train.py [stock] [window] [episodes]")

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(
  sys.argv[3])
batch_size = 32
agent = Agent(window_size)

env = SimpleTradeEnv(stock_name, window_size, agent)

for e in range(episode_count + 1):
  print("Episode " + str(e) + "/" + str(episode_count))
  state = env.reset()

  agent.inventory = []
  done = False

  # An episode
  while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
      print("--------------------------------")
      print("Total Profit: " + formatPrice(env.total_profit))
      print("--------------------------------")

    if len(agent.memory) > batch_size:
      agent.expReplay(batch_size)

  if e % 10 == 0:
    agent.model.save("models/model_ep" + str(e))
