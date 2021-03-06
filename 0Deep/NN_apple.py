import numpy as np

def softmax(a):#ソフトマックス関数
    c = np.max(a)
    exp_a = np.exp(a - c)#オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

def cross_entropy_error(y, t):#交差エントロピー誤差
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) /batch_size

class MulLayer:#乗算レイヤ
    def __init__(self):#xとyの初期化
        self.x = None
        self.y = None
        
    def forward(self, x, y):#順伝播
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):#逆伝播
        dx = dout * self.y #xとyをひっくり返す
        dy = dout * self.x
        
        return dx, dy
    
class AddLayer:#加算レイヤ
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
class ReLU:#ReLU関数
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
class Sigmoid:#シグモイド関数
    def __init__ (self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx

class Affine:#アフィンレイヤ
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx
 
class SoftmaxWithLoss:#ソフトマックス、交差エントロピー誤差レイヤ
    def __init__(self):
        self.loss = None#損失
        self.y = None#softmaxの出力
        self.t = None#教師データ
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
       
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
    
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)#(1)
orange_price = mul_orange_layer.forward(orange, orange_num)#(2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)#(3)
price = mul_tax_layer.forward(all_price, tax)#(4)

#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)#(4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)#(3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)#(2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)#(1)

print(price)#715
print(dapple_num, dapple, dorange, dorange_num, dtax) #110 2.2 3.3 165 650