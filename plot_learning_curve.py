import sys
import pandas as pd
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from tabulate import tabulate

if len(sys.argv) != 3:
  print("Usage: python plot_learning_curve.py [config] [stock]")
  exit()

config_file = sys.argv[1]
stock_name = sys.argv[2]

with open(config_file) as f:
  yaml = YAML()
  config = yaml.load(f)

result_dir = config["result_dir"]

df = pd.read_csv(result_dir + "/learning_curve_{}.csv".format(stock_name))
print(tabulate(df, headers='keys', tablefmt='psql'))

xs = df["Episode"].values
ys = df["Total profit"].values

plt.figure()
plt.plot(xs, ys)
plt.xlabel("Episodes")
plt.ylabel("Total profit")
plt.show()
