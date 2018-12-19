from agent.agent import Agent
from functions import *
import sys
from ruamel.yaml import YAML
from environment import SimpleTradeEnv

with open(sys.argv[1]) as f:
  yaml = YAML()
  config = yaml.load(f)

stock_name = config["stock_name"]
window_size = config["window_size"]
episode_count = config["episode_count"]
result_dir = config["result_dir"]
batch_size = config["batch_size"]

agent = Agent(window_size,
              learning_rate=config["learning_rate"],
              gamma=config["gamma"])

# Environment
env = SimpleTradeEnv(stock_name, window_size, agent,
                     clip_reward=config["clip_reward"],
                     reward_for_buy=config["reward_for_buy"])

# Loop over episodes
for e in range(episode_count + 1):
  # Initialization before starting an episode
  print("Episode " + str(e) + "/" + str(episode_count))
  state = env.reset()
  agent.inventory = []
  done = False

  # Loop in an episode
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
    agent.model.save(result_dir + "/model_ep" + str(e))
