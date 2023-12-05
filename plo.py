import trunk as t
import numpy as np
import matplotlib.pyplot as plt

flag = np.load("result.npy",allow_pickle=True)

blocks = np.unique(flag)

t.plot(flag)