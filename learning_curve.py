import os, re, sys
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from tabulate import tabulate
from agent.agent import Agent
from environment import SimpleTradeEnv
from functions import formatPrice

def get_window_size(model_name):
  """TODO: Refactor this code wrt window_size"""
  model = load_model("models/" + model_name)
  window_size = model.layers[0].input.shape.as_list()[1]

  return window_size

def eval_model(stock_name, model_name):
  # Agent
  window_size = get_window_size(model_name) - 1
  agent = Agent(window_size, True, model_name)

  # Environment
  env = SimpleTradeEnv(stock_name, window_size, agent, print_trade=False)

  # Main loop
  state = env.reset()
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

  keras.backend.clear_session()

  return env.total_profit

def get_num_ep(model_name):
  # https://stackoverflow.com/questions/1327369
  s = re.search("model_ep([0-9]{1,3})", model_name)
  if s:
    return int(s.group(1))
  else:
    return None

def main(stock_name):
  model_names = os.listdir("./models")

  eps = []
  tprofits = []

  # Sort model name in ascending order
  eps = [get_num_ep(n) for n in model_names]
  ixs = np.argsort(eps)
  model_names = [model_names[ix] for ix in ixs]

  # Evaluate all models
  for model_name in model_names:
    total_profit = eval_model(stock_name, model_name)
    print("{:15s} total profit = ".format(model_name) +
          formatPrice(total_profit))
    tprofits.append(total_profit)

  df = pd.DataFrame({"Episode": eps, "Total profit": tprofits})
  df.to_csv("logs/result.csv", index=False)
  print(tabulate(df, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python learning_curve.py [stock]")

  stock_name = sys.argv[1]
  main(stock_name)
