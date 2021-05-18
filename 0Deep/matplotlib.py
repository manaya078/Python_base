import numpy as np
import matplotlib.pyplot as plt

#データの作成
x = np.arange(0, 6, 0.1) #0から6まで0.1刻みで生成
y1 = np.sin(x)
y2 = np.cos(x)

#グラフの描画
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos")#破線で描画
plt.xlabel("x")#x軸のラベル
plt.ylabel("y")#y軸のラベル
plt.title('sin & cos')#タイトル
plt.legend()#要素
plt.show()