import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from Perceptron import Perceptron

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.25,
                           random_state=0)

y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.


perceptron = Perceptron()
perceptron.fit(X, y)  # Ajusta els pesos
y_prediction = perceptron.predict(X)  # Prediu


#  Resultats
bias = perceptron.w_[0]
w1, w2 = perceptron.w_[1:]

m = -w1 / w2
c = -bias / w2

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x_vals = np.linspace(x_min, x_max, 100)
y_vals = m * x_vals + c

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y_prediction, cmap="bwr", alpha=0.7)
plt.plot(x_vals, y_vals, 'k--', label="Límite de decisión")
plt.legend()
plt.savefig("resultado.png")  # Guarda el gráfico
