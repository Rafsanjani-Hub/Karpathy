from engine import Value
from visualization import connect_dots
import gradients_check # gradients_check.sanity_check_3()

import numpy as np
import torch

# np.random.seed(101)

class Module:
    def zero_grad(self,):
        for parameter in self.parameters():
            parameter.grad = 0.0
        #end-def
    #end-def
#end-def

class Neuron(Module):
    def __init__(self, n_inputs, nonlinearity=True):
        self.weights = [Value(data=np.random.uniform(-1, 1)) * (n_inputs ** (-0.5)) for _ in range(n_inputs)]
        self.bias    = Value(data=0.0)
        self.nonlinearity = nonlinearity
    #end-def

    def __call__(self, xs):
        F = sum([weight*x for  weight, x in zip(self.weights, xs)]) + self.bias
        return F.relu() if self.nonlinearity == True else F
    #end-def

    def parameters(self):
        return self.weights + [self.bias]
    #end-def
#end-def

class Layer(Module):
    def __init__(self, n_inputs, n_neurons, nonlinearity):
        self.neurons = [Neuron(n_inputs=n_inputs, nonlinearity=nonlinearity) for _ in range(n_neurons)]
    #end-def

    def __call__(self, xs):
        return [neuron(xs) for neuron in self.neurons]
    #end-def

    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]
    #end-def
#end-def

class MLP(Module):
    def __init__(self, n_inputs, ns_neurons):
        ns_neurons = [n_inputs] + ns_neurons
        self.layers = [Layer(ns_neurons[i], ns_neurons[i+1], nonlinearity=(i!=len(ns_neurons)-2)) for i in range(len(ns_neurons)-1)]
    #end-def

    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        #end
        return xs[0] if len(xs) == 1 else xs
    #end-def

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]
    #end-def
#end-def
