import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_csv("logs/result.csv")
print(tabulate(df, headers='keys', tablefmt='psql'))

xs = df["Episode"].values
ys = df["Total profit"].values

plt.figure()
plt.plot(xs, ys)
plt.show()
