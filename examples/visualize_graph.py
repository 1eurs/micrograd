import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.engine.value import Value
from src.utils.visualize import draw_dot

# Simple computation graph
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a * b; c.label = 'c'
d = c + Value(1.0, label='one'); d.label = 'd'
d.backward()

# Visualize
draw_dot(d).render('computation_graph', format='svg', view=True)