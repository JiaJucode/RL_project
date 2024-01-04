import pandas as pd
import matplotlib.pyplot as plt
import sys

args = sys.argv
image_path = "error_plot.jpg"
data_path = "tf_agents_files/log_files/"
# data_path = "/te.txt"
if len(args) == 3:
    path = args[1]
    data_path = args[2]

returns = pd.Series([float(line) for line in open(data_path + "returns.txt", 'r')])
critic_loss = pd.Series([float(line) for line in open(data_path + "critic_loss.txt", 'r')])
actor_loss = pd.Series([float(line) for line in open(data_path + "actor_loss.txt", 'r')])
returns.plot()
plt.title("returns")
plt.savefig(path + "returns.jpg")
critic_loss.plot()
plt.title("critic_loss")
plt.savefig(path + "critic_loss.jpg")
actor_loss.plot()
plt.title("actor_loss")
plt.savefig(path + "actor_loss.jpg")
