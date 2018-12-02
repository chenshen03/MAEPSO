#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : CHEN Shen

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def Tablet(x):
    sum = np.power(10, 6) * np.square(x[0])
    sum += np.sum(np.square(x[1:]))
    return sum


def Quadric(x):
    sum = 0
    for i in range(len(x)):
        sum += np.square(np.sum(x[:i+1]))
    return sum


def Rosenbrock(x):
    sum = 0
    for i in range(len(x)-1):
        sum += (100 * np.square((x[i+1] - x[i] * x[i])) + np.square(x[i] - 1))
    return sum


def Griewank(x):
    sum = np.sum(x * x) / 4000
    prod = 1
    for i in range(len(x)):
        prod = prod * np.cos(x[i]/np.sqrt(i+1))
    return sum - prod + 1


def Rastrigin(x):
    return np.sum(x * x - 10 * np.cos(2 * np.pi * x) + 10)


def Schaffer(x):
    sum_square = np.sum(np.square(x))
    fact1 = np.sin(np.sqrt(sum_square)) - 0.5
    fact2 = np.square((1 + 0.001 * sum_square))
    return 0.5 + fact1 / fact2


def SchafferF7(x):
    import math
    return sum([(x[i] ** 2 + x[i + 1] ** 2) ** 0.25 for i in range(len(x) - 1)]) + \
           math.sin((50 * sum([(x[i] ** 2 + x[i + 1] ** 2) ** 0.1 for i in range(len(x) - 1)])) ** 2) + 1


def plot_data(func, W=5):
    X = np.linspace(-W, W, 100)
    Y = np.linspace(-W, W, 100)
    X, Y = np.meshgrid(X, Y)
    row, col = X.shape
    Z = np.zeros((row, col))

    for r in range(row):
        for c in range(col):
            x = X[r][c]
            y = Y[r][c]
            v = np.array([x, y])
            Z[r][c] = func(v)
    return X, Y, Z


def plot_contour(X, Y, Z):
    plt.figure()
    # 填充等高线
    plt.contourf(X, Y, Z, 20, cmap=plt.cm.hot)
    # 添加等高线
    C = plt.contour(X, Y, Z, 5)
    plt.clabel(C, inline=True, fontsize=12)
    plt.show()


def plot_3Dshape(X, Y, Z):
    fig = plt.figure()
    ax = Axes3D(fig)
    # Customize the z axis.
    ax.set_zlim(0, 5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    X, Y, Z = plot_data(SchafferF7, W=10)
    plot_3Dshape(X, Y, Z)
    plot_contour(X, Y, Z)
