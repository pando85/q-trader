import os, re, sys
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from ruamel.yaml import YAML
from tabulate import tabulate
from agent.agent import Agent
from environment import SimpleTradeEnv
from functions import formatPrice

def get_window_size(model_name, result_dir):
  """TODO: Refactor this code wrt window_size"""
  model = load_model(result_dir + "/" + model_name)
  window_size = model.layers[0].input.shape.as_list()[1]

  return window_size

def eval_model(config, stock_name, model_name):
  # Agent
  agent = Agent(config["window_size"], True, model_name,
                result_dir=config["result_dir"],
                learning_rate=None,
                gamma=config["gamma"],
                optimizer=None)

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

def main(config, stock_name):
  result_dir = config["result_dir"]
  model_names = [f for f in os.listdir(result_dir) if f.startswith("model_ep")]

  tprofits = []

  # Sort model name in ascending order
  eps = [get_num_ep(n) for n in model_names]
  ixs = np.argsort(eps)
  eps = [eps[ix] for ix in ixs]
  model_names = [model_names[ix] for ix in ixs]

  # Evaluate all models
  for model_name in model_names:
    total_profit = eval_model(config, stock_name, model_name)
    print("{:15s} total profit = ".format(model_name) +
          formatPrice(total_profit))
    tprofits.append(total_profit)

  df = pd.DataFrame({"Episode": eps, "Total profit": tprofits})
  df.to_csv(result_dir + "/learning_curve_{}.csv".format(stock_name), index=False)
  print(tabulate(df, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python learning_curve.py [config] [stock]")

  config_file = sys.argv[1]
  stock_name = sys.argv[2]

  with open(sys.argv[1]) as f:
    yaml = YAML()
    config = yaml.load(f)

  main(config, stock_name)
