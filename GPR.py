# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['font.serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class GPR:
    def __init__(self, X, y, v0 = .04, v1 = 0., alpha = 2., r = .25):
        self.X, self.y = X, y
        self.d = X.shape[1]
        self.v0 = v0
        self.v1 = v1
        self.alpha = alpha
        self.r = np.array(r)
        if(self.r.size == 1):
            self.r = self.r.repeat(self.d)
        
        def loss(params):
            self.v0, self.r = params[0], params[1]
            K11 = self.K(self.X, self.X)
            loss = 0.5 * self.y.T.dot(np.linalg.inv(K11)).dot(self.y) + 0.5 * np.linalg.slogdet(K11)[1] + 0.5 * self.X.shape[0] * np.log(2 * np.pi)
            return loss.ravel()

        res = minimize(loss, [self.v0, self.r], bounds=((1e-4, 1e4), (1e-4, 1e4)), method='L-BFGS-B')
        self.v0, self.r = res.x[0], res.x[1]

    def predict(self, X_star):
        K11 = self.K(self.X, self.X)
        K22 = self.K(X_star, X_star)
        K12 = self.K(self.X, X_star)
        K21 = K12.T
        IK11 = np.linalg.inv(K11)
        
        ystar = K21.dot(IK11).dot(self.y)
        cov = K22 - K21.dot(IK11).dot(K12)
        std = np.sqrt(np.diag(cov))

        return ystar, std

    def K(self, x1, x2):
        A = np.abs(x1.reshape(-1, 1, self.d) - x2.reshape(1, -1, self.d))
        return self.v0 * np.exp(-np.sum(A**self.alpha / self.r.reshape(1, 1, self.d), 2) / 2.) + self.v1

X = np.random.rand(20, 1) * 10
Xstar = np.arange(0, 10, 0.1).reshape(-1, 1)
z = lambda X: np.cos(X)
sigma = 1e-4

n = np.random.normal(0, sigma, X.shape)
y = z(X) + n

MyGPR = GPR(X, y)
print(MyGPR.v0, MyGPR.r)
ystar, std = MyGPR.predict(Xstar)
test_y = ystar.squeeze()
plt.figure()
plt.scatter(X, y, color='red', label='样本点')
plt.plot(Xstar, z(Xstar), '--r', label='真实曲线')
plt.plot(Xstar, test_y, color='black', label='预测曲线')
plt.plot(Xstar, test_y + std, ':', color='black', linewidth=0.8)
plt.plot(Xstar, test_y - std, ':', color='black', linewidth=0.8)
plt.fill_between(Xstar.ravel(), test_y + std, test_y - std, alpha=0.1, color='cyan', label='一倍标准差')
plt.grid('on')
plt.legend()
plt.show()