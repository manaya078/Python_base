def AND(x1, x2):#ANDの論理回路
    w1, w2, theta = 0.5, 0.5, 0.7#重み1と2、閾値の初期化
    tmp = x1*w1 + x2*w2#重み付き入力の総和
    if tmp <= theta:#tmpが閾値以下なら
        return 0
    elif tmp > theta:#tmpが閾値より大きいなら
        return 1
    
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

def NAND(x1, x2):#NANDの論理回路(ANDの重みと閾値の正負を逆に)
    w1, w2, theta = -0.5, -0.5, -0.7#重み1と2、閾値の初期化
    tmp = x1*w1 + x2*w2#重み付き入力の総和
    if tmp <= theta:#tmpが閾値以下なら
        return 0
    elif tmp > theta:#tmpが閾値より大きいなら
        return 1

print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))

def OR(x1, x2):#ORの論理回路
    w1, w2, theta = 0.5, 0.5, 0#重み1と2、閾値の初期化
    tmp = x1*w1 + x2*w2#重み付き入力の総和
    if tmp <= theta:#tmpが閾値以下なら
        return 0
    elif tmp > theta:#tmpが閾値より大きいなら
        return 1
    
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))