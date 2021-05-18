import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('buiikikaesu.jpg')#画像の読み込み
plt.imshow(img)

plt.show()