import numpy as np
import matplotlib.pyplot as plt

# データを作成する。
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


x = np.linspace(0, 5, 300)
y = np.linspace(0, 5, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contourf(X,Y,Z,100, cmap="coolwarm")
plt.show()