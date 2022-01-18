from contextlib import contextmanager
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# @contextmanager
class visualization(object):
    def __init__(self, axsize=(1,1), figsize=(8,5)):
        super().__init__()
        self.fig, self.axes = plt.subplots(*axsize, figsize=figsize)
    
    def __enter__(self):
        plt.ion()
        plt.show() # 实时显示，动态更新
        return self, self.fig, self.axes
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.legend()
        plt.ioff()
        plt.show() # 保持显示
    
    def set_data(self, x, mu, std):
        self.x = x
        self.mu = mu
        self.std = std

    def plot_confidence_interval(self, ax, alpha):
        conf_intveral = stats.norm.interval(alpha, loc=self.mu, scale=np.clip(self.std, a_min=1e-8, a_max=None))
        ax.fill_between(self.x, conf_intveral[0], conf_intveral[1], alpha=0.1, label=f"{alpha*100:.1f}% confidence interval")

    def plot_predict_result(self, ax):
        ax.plot(self.x, self.mu, label="predict", linewidth=2)
