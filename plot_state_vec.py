import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from agent.agent import Agent
from environment import SimpleTradeEnv

def take_sample_states(stock_name, window_size):
  # Agent
  agent = Agent(window_size, is_eval=False, model_name=None)

  # Environment
  env = SimpleTradeEnv(stock_name, window_size, agent)

  # Main loop
  state = env.reset()
  states = []
  states.append(state)
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    states.append(state)

  keras.backend.clear_session()

  states = [s[0, :-1] for s in states]

  return states

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python learning_curve.py [stock] [window_size]")

  stock_name = sys.argv[1]
  window_size = int(sys.argv[2])
  samples = take_sample_states(stock_name, window_size)
  print("{} samples".format(len(samples)))

  plt.figure()

  for s in samples[:2000:10]:
    plt.plot(np.arange(window_size), s)

  plt.show()
