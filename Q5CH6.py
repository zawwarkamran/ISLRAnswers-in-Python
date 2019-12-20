import pandas as pd
import numpy as np 

data = np.random.normal(size=100)
noise = np.random.normal(size=100)

reponse  = np.sum(0.1 + 0.2*data + 0.3*data^2 + 0.4*data^3 + noise)
print(reponse)