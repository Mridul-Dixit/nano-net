import math

class Tensor:
    def __init__(self, _data, _children=()):
        self.data = _data
        self.children = set(_children)
        self.grad = 0
        self._backward = lambda:None
    
    def __repr__(self):
        return f"Tensor(data = {self.data}, grad = {self.grad}) "
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self,other):
        assert isinstance(other,(int, float)), "Power should be int or float"
        out = Tensor(self.data ** other,(self,))
        
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(math.exp(self.data),(self,))
        
        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Tensor((2 * self.exp().data - 1) / (2 * self.exp().data + 1),(self,))
        
        def _backward():
            self.grad += 1 - (out.data) ** 2
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(max(0, self.data),(self,))
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
        
    def sigmoid(self):
        if self.data > 0:
            exp_neg_x = Tensor((-self).exp().data, (self,))
            out = Tensor(1 / (1 + exp_neg_x.data), (self,))
            
        else: 
            exp_x = self.exp()
            out = Tensor(exp_x.data / (1 + exp_x.data), (self,))
            
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out

    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other): 
        return other * self**-1
        
    def backward(self):
        top_order = []
        visited = set()
        def top_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    top_sort(child)
                top_order.append(v)
        top_sort(self)
        self.grad = 1
        for node in reversed(top_order):
            node._backward()