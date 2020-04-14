import csv
from urllib.request import urlopen

import pandas as pd

from env import MultiStockEnv

# Get stocks
STOCKS_URL = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/aapl_msi_sbux.csv'
response = urlopen(STOCKS_URL)
data = pd.read_csv(response).values

env = MultiStockEnv(data)
print(env.action_list)