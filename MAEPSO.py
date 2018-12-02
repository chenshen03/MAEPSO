#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : CHEN Shen

import sys
from functions import *


class MAEPSO(object):
    """
    MAEPSO算法
    陶新民, 刘福荣, 刘玉,等. 一种多尺度协同变异的粒子群优化算法[J]. 软件学报, 2012, 23(7):1805-1815.
    """

    def __init__(self, func=lambda x:np.sum(np.square(x), axis=1), methods='MAEPSO'):
        """
        算法参数设置
        :param func: 适应度函数
        """
        self.size = 20                      # 粒子群数量
        self.M = 5                          # 尺度个数
        self.dim = 30                       # 函数维度
        self.W = 100                        # 变量空间的宽度
        self.c1 = 1.2                       # 个体最优值的学习因子
        self.c2 = 1.2                       # 全局最优值的学习因子
        self.w_max = 0.8                    # 最大惯性因子
        self.w_min = 0.4                    # 最小惯性因子
        self.w = self.w_max                 # 惯性因子
        self.P = (int)(self.size / self.M)  # 每个子群的粒子个数
        self.func = func                    # 适应度函数
        self.methods = methods              # 粒子群算法
        self._init_PSO()

    def _init_PSO(self):
        """
        位置、速度等变量初始化
        :return:
        """
        self.X = np.random.uniform(-self.W, self.W, (self.size, self.dim))  # 初始化粒子位置
        self.pb = np.copy(self.X)                                           # 个体最优位置
        self.gb = self.X[np.argmin(self.f(self.X))]                         # 全局最优位置
        self.V = np.random.uniform(-1, 1, (self.size, self.dim))            # 初始化粒子速度
        self.T = np.ones(self.dim) * 0.5                                    # 速度的阈值
        self.G = np.zeros(self.dim)                                         # 速度的逃逸次数
        self.sigma = np.ones(self.M) * 2 * self.W                           # M个尺度的高斯变异算子方差

    def f(self, X):
        """
        粒子群的适应度
        :param X: 粒子群
        :return: 适应度
        """
        if X.ndim == 1:
            return self.func(X)
        return [self.func(x) for x in X]

    def FitX(self, m):
        """
        计算第m个子群的总适应度
        :param m: 子群索引
        :return: 子群的总适应度
        """
        sub_group = self.X[(m-1)*self.P:m*self.P]
        sub_f = self.f(sub_group)
        return np.sum(sub_f) / self.P

    def update_sigma(self):
        """
        更新所有高斯变异算子的方差
        :return:
        """
        # ind = np.argsort(self.f((self.X)))
        # self.X = self.X[ind]
        # self.V = self.V[ind]
        FitXs = [self.FitX(m+1) for m in range(self.M)]
        max_FitX = np.max(FitXs)
        min_Fitx = np.min(FitXs)
        total_FitX = np.sum(FitXs)

        for i in range(self.M):
            self.sigma[i] *= np.exp((self.M * FitXs[i] - total_FitX) / (max_FitX - min_Fitx + pow(10, -10)))

            while self.sigma[i] > self.W / 4:
                self.sigma[i] -= self.W / 4
        # print(FitXs)
        # print(self.sigma)

    def update_T(self):
        """
        自适应阈值设定
        :return:
        """
        k1 = 5
        k2 = 10
        ind = self.G > k1
        self.T[ind] /= k2
        self.G[ind] = 0

    def update_VP(self, methods='MASPSO'):
        """
        更新粒子群的速度和位置
        :return:
        """
        V = self.V
        X = self.X

        for i in range(self.size):
            for j in range(self.dim):

                # 速度更新公式
                V[i][j] = self.w * V[i][j] + \
                                  self.c1 * np.random.rand() * (self.pb[i][j] - X[i][j]) + \
                                  self.c2 * np.random.rand() * (self.gb[j] - X[i][j])

                # 判断是否需要逃逸, 若满足逃逸条件，则进行逃逸
                if np.abs(V[i][j]) < self.T[j] and self.methods == 'MAEPSO':
                    # 第j个速度的逃逸次数加1
                    self.G[j] += 1

                    min_f = sys.maxsize
                    min_ind = 0
                    temp_x = X[i][j]
                    randn_sigma = np.random.randn(self.M) * self.sigma
                    for m in range(self.M):
                        X[i][j] = temp_x + randn_sigma[m]
                        temp_f = self.f(X[i])
                        if temp_f < min_f:
                            min_f = temp_f
                            min_ind = m

                    Vmax = self.W - abs(X[i][j])
                    rand = np.random.uniform(-1, 1)
                    X[i][j] = temp_x + rand * Vmax
                    if min_f < self.f(X[i]):
                        V[i][j] = randn_sigma[min_ind]
                    else:
                        V[i][j] = rand * Vmax

                    X[i][j] = temp_x

                # 更新粒子的位置
                X[i][j] += V[i][j]
                # 更新个体经历过的最优位置
                if self.f(self.pb[i]) > self.f(X[i]):
                    self.pb[i] = X[i]
                # 更新全局最优位置
                if self.f(self.gb) > self.f(self.pb[i]):
                    self.gb = self.pb[i]

        self.V = V
        self.X = X

    def evolve(self, steps=1000, trials=1):
        """
        粒子群进化过程
        :param steps: 迭代次数
        :param trials: 实验次数
        :return:
        """
        X, Y, Z = plot_data(self.func, self.W)
        w_max = self.w_max
        w_min = self.w_min
        results = np.array([])
        print(self.func.__name__)
        for t in range(trials):
            self._init_PSO()
            plt.figure()
            fitness_iter = np.array([])
            for s in range(steps):
                self.update_VP()
                self.update_sigma()
                self.update_T()
                # HPSO论文(3)，消除速度惯性部分的公式
                self.w = w_max - (w_max - w_min) * s / steps

                plt.clf()
                plt.contour(X, Y, Z, 20)
                plt.scatter(self.X[:, 0], self.X[:, 1], s=30, color='k')
                plt.xlim(-self.W, self.W)
                plt.ylim(-self.W, self.W)
                plt.pause(0.01)

                print("iter:{} best fitness: {}, mean fitness: {}".format(s+1, self.f(self.gb), np.mean(self.f(self.pb))))
                if self.f(self.gb) == 0:
                    print("finished")
                    break
                fitness_iter = np.append(fitness_iter, self.f(self.gb))

            # if t == 0:
            #     # 第一次测试时的收敛情况曲线图
            #     plt.plot(np.log10(fitness_iter))
            #     plt.title(self.func.__name__)
            #     plt.xlabel("Iteration number N")
            #     plt.ylabel("Log(function fitness)")
            #     plt.xlim(0, 6000, 500)
            #     plt.savefig("./results/" + self.func.__name__ + ".jpg")

            res = self.f(self.gb)
            results = np.append(results, res)
            print("trial {}: final fitness: {}".format(t+1, res))

        results = np.array(results)
        print("min fitness: {}".format(np.min(results)))
        print("median fitness: {}".format(np.median(results)))
        print("varience fitness: {}".format(np.var(results)))
        print("mean fitness: {}".format(np.mean(results)))
        print("max fitness: {}".format(np.max(results)))
        print("")


if __name__ == '__main__':
    pso = MAEPSO(func=Tablet, methods='MAEPSO')
    pso.evolve(6000, 50)