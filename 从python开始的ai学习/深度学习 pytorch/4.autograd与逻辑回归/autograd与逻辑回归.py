import torch
torch.manual_seed(10)


# ==================== retain_graph ==================== #
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad = True)
    x = torch.tensor([2.], requires_grad = True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph = True)
    # print(w.grad)
    y.backward()

# ==================== grad_tensors ==================== #
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)    # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)
    y1 = torch.add(a, b)

    loss = torch.cat([y0, y1], dim = 0)
    grad_tensors = torch.tensor([1, 1])

    loss.backward(gradient=grad_tensors)

    print(w.grad)

# ==================== autograd.grad ==================== #
# flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)    # y = x ** 2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)
    print(grad_1)  # grad_1 是个元组

    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)

# ==================== Tips1 ==================== #
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        w.grad.zero_()  # 手动对梯度清零,_表示原位(in-place)操作

# ==================== Tips2 ==================== #
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)


# ==================== Tips3 ==================== #
flag = True
# flag = False
if flag:

    a = torch.ones((1,))
    print(id(a), a)

    a = a + torch.ones((1, ))
    print(id(a), a)

    # a += torch.ones((1, ))
    # print(id(a), a)

# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)

    y.backward()