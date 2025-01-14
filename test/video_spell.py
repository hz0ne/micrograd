
from micrograd.engine import Value
from micrograd.trace import draw_dot

# derivative and bias deviate

def lol():
    # 展示一点导数的数学逻辑，取 h 无限趋近于0
    h = 0.01

    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    e = a * b
    d = e + c
    f = Value(-2.0)
    L = d * f
    L1 = L.data

    a = Value(2.0 + h)
    b = Value(-3.0)
    c = Value(10.0)
    e = a*b
    d = e + c
    f = Value(-2.0)
    L = d * f
    L2 = L.data

    print(L2-L1 / h)

def draw_input_topo():


    # inputs
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of the nuron
    b = Value(6.88137, label='b')

    x1w1 = x1 * w1
    x1w1.label = 'x1w1'
    x2w2 = x2 * w2
    x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b
    n.label = 'n'

    o = n.tanh()
    o.label = 'o'

    # 链式求导
    # o.grad = 1 最后结果一定是1
    # n.grad = do/dn = 1-tanh(n)**2 = 1 - o**2 = 0.5
    # b.grad = do/db = do/dn * dn/db = 0.5 * 1， 链式求导，常数流向
    # (x1w1 + x2w2).grad = 0.5, 常数流向
    # x1w1.grad = 0.5
    # x2w2.grad = 0.5

    o.backward()

    obj = draw_dot(o)
    obj.render( view=True)




if __name__ == '__main__':
    draw_input_topo()