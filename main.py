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

with visualization(axsize=(3,1)) as (vis, fig, axes):
    vis.plot_predict_result(axes[0])
    vis.plot_confidence_interval(axes[0], 0.95)
    axes[0].plot(data['Close'], label='Close')
    axes[0].set_title(", ".join([param + "=" + f"{gpr.params[param]:.2f}" for param in gpr.params]))
    axes[0].grid('on')