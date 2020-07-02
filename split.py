import os
import pandas as pd

path = os.path.abspath(os.path.join(os.path.dirname(__file__), './data'))

plant1 = pd.read_csv(path + '/plant1_test_split.csv')
plant1.index = plant1['index']
print(plant1)

# plant1.to_csv(path + '/plant1_test_split.csv')