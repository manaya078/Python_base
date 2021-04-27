"""
シグモイド関数

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
"""

import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):#シグモイド関数
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)#-5.0から5.0まで0.1刻み
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)#y軸の範囲
plt.show()