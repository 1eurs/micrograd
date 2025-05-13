class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError("Optimizer step not implemented")

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad