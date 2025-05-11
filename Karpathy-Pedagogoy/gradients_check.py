import torch
from engine import Value

def sanity_check_1():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

    print('It perfectly works.')
#end-def

def sanity_check_2():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

    print('It perfectly works.')
#end-def

def sanity_check_3():

    # micrograds:
    a = Value(-2.0, label='a')
    b = Value(3.0, label='a')

    d = a * b
    e = a + b

    f = e * d

    f.backward()
    
    f_data_mg = f.data
    a_grad_mg = a.grad
    b_grad_mg = b.grad

    # using PyTorch:
    a = torch.tensor(-2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    d = a * b
    e = a + b

    f = e * d
    f.backward()

    f_data_pt = f.data
    a_grad_pt = a.grad
    b_grad_pt = b.grad

    #Forward pass well well.
    assert f_data_mg == f_data_pt.item()

    #Backward pass well well.
    assert a_grad_mg == a_grad_pt.item()
    assert b_grad_mg == b_grad_pt.item()

    print('It perfectly works.')
#end-def
