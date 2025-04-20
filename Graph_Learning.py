import math
from typing import List
import random

# Micrograd's Value class for autograd
class Value:
    def __init__(self, data: float, _children=(), _op: str=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Base Module class
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> List[Value]:
        return []

# Graph Convolutional Layer
class GCNLayer(Module):
    def __init__(self, in_features: int, out_features: int):
        self.weight = [[Value(random.uniform(-1, 1)) for _ in range(in_features)] for _ in range(out_features)]
        self.bias = [Value(0.0) for _ in range(out_features)]

    def __call__(self, x: List[List[Value]], adj: List[List[float]]) -> List[List[Value]]:
        n = len(x)  # number of nodes
        out = [[Value(0.0) for _ in range(len(self.weight))] for _ in range(n)]
        
        # Degree normalization (D^-1/2 * A * D^-1/2) with self-loops
        deg = [sum(row) + 1 for row in adj]  # Add 1 for self-loop
        adj_norm = [[Value((adj[i][j] + (1 if i == j else 0)) / math.sqrt(deg[i] * deg[j]) if deg[i] * deg[j] > 0 else 0.0) for j in range(n)] for i in range(n)]
        
        # GCN: A_norm * X * W + b
        for i in range(n):
            for k in range(len(self.weight)):
                s = self.bias[k]
                for j in range(n):
                    for l in range(len(self.weight[0])):
                        s += adj_norm[i][j] * x[j][l] * self.weight[k][l]
                out[i][k] = s.relu()
        return out

    def parameters(self) -> List[Value]:
        return [p for row in self.weight for p in row] + self.bias

# GNN Model
class GNN(Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def __call__(self, x: List[List[Value]], adj: List[List[float]]) -> List[List[Value]]:
        h = self.layer1(x, adj)
        return self.layer2(h, adj)

    def parameters(self) -> List[Value]:
        return self.layer1.parameters() + self.layer2.parameters()

# Synthetic graph dataset
def create_synthetic_graph(n_nodes: int, n_features: int) -> tuple:
    adj = [[0.0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < 0.3:
                adj[i][j] = adj[j][i] = 1.0
    
    x = [[Value(random.uniform(-1, 1)) for _ in range(n_features)] for _ in range(n_nodes)]
    labels = [random.randint(0, 1) for _ in range(n_nodes)]
    return x, adj, labels

# Training function
def train_gnn():
    n_nodes = 10
    n_features = 4
    hidden_features = 8
    out_features = 2
    lr = 0.01
    epochs = 1501
    
    x, adj, labels = create_synthetic_graph(n_nodes, n_features)
    model = GNN(n_features, hidden_features, out_features)
    
    for epoch in range(epochs):
        model.zero_grad()
        out = model(x, adj)
        
        # Cross-entropy loss (simplified)
        loss = Value(0.0)
        for i in range(n_nodes):
            score = out[i][1] - out[i][0]  # Logit difference
            target = Value(1.0 if labels[i] == 1 else -1.0)
            loss += (score - target) * (score - target)
        loss = loss * (1.0 / n_nodes)
        
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

if __name__ == "__main__":
    train_gnn()
