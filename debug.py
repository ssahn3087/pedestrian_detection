import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import numpy.random as npr


model = TSNE(learning_rate=100)
feature = npr.randn(1000, 4096)
labels = npr.randint(5, size=1000)
embedded_data = model.fit_transform(feature)

xs = embedded_data[:, 0]
ys = embedded_data[:, 1]
plt.scatter(xs, ys, c= labels)

plt.show()