import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize
from visualization import visualization

class GPR:
    def __init__(self, optimize=True, kernel='squared_exp'):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"r": 2.0, "alpha":2.0, "v0": 2.0, "v1": 0.0, "v2": 0.0, "r2": 20.0, "v02": 1.0, "nu":1.0}
        self.optimize = optimize
        self.y_mean = 0
        self.d = 1
        self.kernel = {
            'squared_exp': self.squared_exponential_kernel, 
            'LS_squared_exp': self.LS_squared_exponential_kernel,
            'sin_exp': self.sin_exponential_kernel,
            'decay_sin_exp': self.decay_sin_exponential_kernel
        }[kernel]

    def fit(self, X, y, **optim_range):
        # store train data
        self.y_mean = np.mean(y)
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y) - self.y_mean
        self.d = self.train_X.shape[1]
        for item in list(optim_range):
            if(isinstance(optim_range[item], (int, float))):
                self.params[item] = optim_range.pop(item)

        # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            for i, p in enumerate(optim_range):
                self.params[p] = params[i]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, 
                [self.params[p] for p in optim_range],
                bounds=tuple(optim_range.values()),
                method='L-BFGS-B')
            for i, p in enumerate(optim_range):
                self.params[p] = res.x[i]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y) + self.y_mean
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        std = np.sqrt(np.diag(cov))
        return mu, std

    def squared_exponential_kernel(self, x1, x2):
        A = np.abs(x1.reshape(-1, 1, self.d) - x2.reshape(1, -1, self.d))
        B = A**self.params["alpha"] / np.array(self.params["r"]).reshape(1, 1, self.d)
        return self.params["v0"] * np.exp(-np.sum(B, 2) / 2.) + self.params["v1"] + self.params["v2"] * np.all(x1.reshape(-1, 1, self.d) == x2.reshape(1, -1, self.d), axis=2)

    def LS_squared_exponential_kernel(self, x1, x2):
        A = np.abs(x1.reshape(-1, 1, self.d) - x2.reshape(1, -1, self.d))
        B1 = A**self.params["alpha"] / np.array(self.params["r"]).reshape(1, 1, self.d)
        B2 = A**self.params["alpha"] / np.array(self.params["r2"]).reshape(1, 1, self.d)
        v0 = self.params["v0"] * np.exp(-np.sum(B1, 2) / 2.) + self.params["v02"] * np.exp(-np.sum(B2, 2) / 2.)
        v1 = self.params["v1"]
        v2 = self.params["v2"] * np.all(x1.reshape(-1, 1, self.d) == x2.reshape(1, -1, self.d), axis=2)
        return v0 + v1 + v2
    
    def sin_exponential_kernel(self, x1, x2):
        A = np.abs(x1.reshape(-1, 1, self.d) - x2.reshape(1, -1, self.d))
        B = np.sin(self.params["nu"] * np.pi * A) / np.array(self.params["r"]).reshape(1, 1, self.d)
        return self.params["v0"] * np.exp(-np.sum(B, 2) / 2.) + self.params["v1"] + self.params["v2"] * np.all(x1.reshape(-1, 1, self.d) == x2.reshape(1, -1, self.d), axis=2)

    def decay_sin_exponential_kernel(self, x1, x2):
        A = np.abs(x1.reshape(-1, 1, self.d) - x2.reshape(1, -1, self.d))
        B = np.sin(self.params["nu"] * np.pi * A) / np.array(self.params["r"]).reshape(1, 1, self.d)
        C = A**2 / np.array(self.params["r2"]).reshape(1, 1, self.d)
        v0 = self.params["v0"] * np.exp(-np.sum(C, 2) / 2.) * np.exp(-np.sum(B, 2) / 2.)
        v1 = self.params["v1"]
        v2 = self.params["v2"] * np.all(x1.reshape(-1, 1, self.d) == x2.reshape(1, -1, self.d), axis=2)
        return v0 + v1 + v2

if __name__ == '__main__':
    def y(x, noise_sigma=0.0):
        x = np.asarray(x)
        y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
        return y

    train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
    train_y = y(train_X, noise_sigma=1e-4)
    test_X = np.arange(0, 15, 0.1).reshape(-1, 1)

    gpr = GPR()
    gpr.fit(train_X, train_y)
    mu, std = gpr.predict(test_X)
    with visualization() as (vis, fig, ax):
        vis.set_data(test_X.reshape(-1), mu.reshape(-1), std)
        vis.plot_confidence_interval(ax, 0.95)
        vis.plot_predict_result(ax)
        ax.set_title(f"r={gpr.params['r']:.2f} alpha={gpr.params['alpha']:.2f} v0={gpr.params['v0']:.2f}")
        ax.scatter(train_X, train_y, label="train", c="red", marker="x")
    
    # def y_2d(x, noise_sigma=0.0):
    #     x = np.asarray(x)
    #     y = np.sin(0.5 * np.linalg.norm(x, axis=1))
    #     y += np.random.normal(0, noise_sigma, size=y.shape)
    #     return y

    # train_X = np.random.uniform(-4, 4, (100, 2)).tolist()
    # train_y = y_2d(train_X, noise_sigma=1e-4)

    # test_d1 = np.arange(-5, 5, 0.2)
    # test_d2 = np.arange(-5, 5, 0.2)
    # test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
    # test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]

    # gpr = GPR(optimize=True)
    # gpr.fit(train_X, train_y)
    # mu, cov = gpr.predict(test_X)
    # z = mu.reshape(test_d1.shape)

    # fig = plt.figure(figsize=(7, 5))
    # ax = Axes3D(fig, auto_add_to_figure=False)
    # fig.add_axes(ax)
    # ax.plot_surface(test_d1, test_d2, z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    # ax.scatter(np.asarray(train_X)[:,0], np.asarray(train_X)[:,1], train_y, c=train_y, cmap=cm.coolwarm)
    # ax.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)
    # ax.set_title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    # plt.show()
