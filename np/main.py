import pandas as pd
import numpy as np

df = pd.read_csv("./test.csv", sep=",", low_memory=False)
data = df.to_numpy()
data = np.delete(data, 0, axis = 1)

print(data[:,3:])