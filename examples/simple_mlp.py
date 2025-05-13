import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nn.mlp import MLP
from src.engine.value import Value
from src.optim.optimizer import SGD

# XOR dataset
xs = [[Value(0), Value(0)], [Value(0), Value(1)], [Value(1), Value(0)], [Value(1), Value(1)]]
ys = [Value(0), Value(1), Value(1), Value(0)]

# Initialize MLP
model = MLP(2, [4, 4, 1])
optimizer = SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    loss = Value(0.0)
    for x, y in zip(xs, ys):
        pred = model(x)
        loss += (pred - y) ** 2
    loss = loss / len(xs)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Test predictions
for x in xs:
    pred = model(x)
    print(f"Input: {[xi.data for xi in x]}, Prediction: {pred.data}")