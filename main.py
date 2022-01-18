import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GPR import GPR
from visualization import visualization

pd.set_option('display.notebook_repr_html',False)
data = pd.read_excel(r'grid.xls')[:-1]

data.insert(loc=0, column='Index', value=np.arange(data.shape[0]))

gpr = GPR(optimize=True)
gpr.fit(train_X, train_y, alpha_range=(0.5,2.5), r_range=(1e2, 1e3), v2_range=(1e-4,1))

with visualization() as (vis, fig, ax):
    vis.set_data(data['Index'][:250].values.reshape(-1), mu.reshape(-1), std)
    vis.plot_predict_result(ax)
    vis.plot_confidence_interval(ax, 0.95)
    # ax.plot(data['Open'], label='Open')
    # ax.plot(data['High'], label='High')
    # ax.plot(data['Low'], label='Low')
    # ax.plot(data['Volume'], label='Volume')
    ax.plot(data['Close'][:250], label='Close')
    ax.set_title(f"r={gpr.params['r']:.2f} alpha={gpr.params['alpha']:.2f} v0={gpr.params['v0']:.2f}")