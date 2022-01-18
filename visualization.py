import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def plot_confidence_interval(ax, x, mu, std, alpha):
    conf_intveral = stats.norm.interval(alpha, loc=mu, scale=np.clip(std, a_min=1e-8, a_max=None))
    ax.fill_between(x, conf_intveral[0], conf_intveral[1], alpha=0.1, label=f"{alpha*100:.1f}% confidence interval")

def plot_predict_result(ax, x, mu):
    ax.plot(x, mu, label="predict")
