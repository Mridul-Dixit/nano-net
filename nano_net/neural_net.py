import random
from nano_net.tensor import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            
    def parameters(self):
        return []

class Neuron:
    def __init__(self,nin, _activation=None):
        self.activation_functions = {
            "relu": lambda x: x.relu(),
            "sigmoid": lambda x: x.sigmoid(),
            "tanh": lambda x: x.tanh(),
        }

        assert _activation in (None, *self.activation_functions.keys()), f"Only {', '.join(self.activation_functions.keys())} activation functions are supported"

        self.weights = [Tensor(random.uniform(1,-1)) for _ in range(nin)]
        self.activation = _activation
        self.bias = Tensor(0)

    def __call__(self, X):
        wei_sum = sum((wi* xi for wi, xi in zip(self.weights, X)), self.bias)
        return self.activation_functions[self.activation](wei_sum) if self.activation else wei_sum

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        activation_name = self.activation.capitalize() if self.activation else "Linear"
        return f"{activation_name}Neuron({len(self.weights)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def forward(self, X):
        out =  [n(X) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts, activations=None):
        sz = [nin] + nouts
        if activations is None:
            activations = ['relu'] * (len(nouts) - 1) + [None]
        assert len(activations) == len(nouts), "Number of activations must match the number of layers."
        self.layers = [
            Layer(sz[i], sz[i + 1], _activation=activations[i])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layer_details = [
            f"{layer} (activation={layer.neurons[0].activation or 'Linear'} )\n"
            for layer in self.layers
        ]
        return f"MLP of [{', '.join(layer_details)}]"
