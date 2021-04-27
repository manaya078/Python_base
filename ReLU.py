"""
ReLU関数(Rectified Linear Unit)
入力が0より大きいなら入力をそのまま出力し、0以下なら0を出力する

def relu(x):
    return np.maximum(0, x)
"""

import numpy as np
import matplotlib.pylab as plt

def relu(x):#ReLU関数
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)#-5.0から5.0まで0.1刻み
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 5.0)#y軸の範囲
plt.show()