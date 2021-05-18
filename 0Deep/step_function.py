"""
パーセプトロンは活性化関数としてステップ関数を利用している

def step_function(x):
   if x > 0:
       return 1
    else:
       return 0
これではxは実数しかとれない(Numpyの配列は引数として使えない)

def step_function(x):
    y = x > 0
    return y.astype(np.int)

yをbool型からint型とすることでtrueが1falseが0に変換される
"""
import numpy as np
import matplotlib.pylab as plt

def step_function(x):#ステップ関数
    return np.array(x > 0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)#-5.0から5.0まで0.1刻み
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)#y軸の範囲
plt.show()