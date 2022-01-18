import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize
from visualization import visualization

class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            return loss.ravel()

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

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
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        std = np.sqrt(np.diag(cov))
        return mu, std

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)


if __name__ == '__main__':
    def y(x, noise_sigma=0.0):
        x = np.asarray(x)
        y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
        return y.tolist()

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
