import os, re, sys
import numpy as np
import pandas as pd
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
  window_size = get_window_size(model_name)
  agent = Agent(window_size, True, model_name)

  # Environment
  env = SimpleTradeEnv(stock_name, window_size, agent)

  # Main loop
  state = env.reset()
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

  print("--------------------------------")
  print("Total Profit: " + formatPrice(env.total_profit))
  print("--------------------------------")

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

  for model_name in model_names:
    ep = get_num_ep(model_name)
    total_profit = eval_model(stock_name, model_name)
    eps.append(ep)
    tprofits.append(total_profit)

  ixs = np.argsort(eps)
  eps = [eps[ix] for ix in ixs]
  tprofits = [tprofits[ix] for ix in ixs]

  df = pd.DataFrame({"Episode": eps, "Total profit": tprofits})
  df.to_csv("logs/result.csv", index=False)
  print(tabulate(df, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python learning_curve.py [stock]")

  stock_name = sys.argv[1]
  main(stock_name)
