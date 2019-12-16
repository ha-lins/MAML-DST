import numpy as np
import matplotlib.pyplot as plt

size = 5
x = np.arange(size)
a = [0.087, 0.2111, 0.1941, 0.1075, 0.3446]
b = [0.5437, 0.5115, 0.5201, 0.5222, 0.4515]

total_width, n = 0.85, 3
width = total_width / n
x = x - (total_width - width) / 2
labels = ['Hotel', 'Train', 'Attraction', 'Restaurant', 'Taxi']
plt.bar(x, a,  width=width, label='Ours w/o DND', facecolor='#1E90FF', edgecolor='white', tick_label=labels)
plt.bar(x + width, b, width=width, label='Ours', facecolor='#ff9999', edgecolor='white', tick_label=labels)
plt.legend()
plt.show()