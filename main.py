import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GPR import GPR
from visualization import visualization

pd.set_option('display.notebook_repr_html',False)
data = pd.read_excel(r'grid.xls')[:-1]
data = pd.read_csv(f"CELYAD ONCOLOGY SA (06-19-2015 _ 01-14-2022).csv")
data.insert(loc=0, column='Index', value=np.arange(data.shape[0]))

train_X = data['Index'][:1000:4].values.reshape(-1, 1)
train_y = data['Close'][:1000:4].values.reshape(-1, 1)

# train_X = np.concatenate((train_X - np.random.rand(*train_X.shape), train_X, train_X + np.random.rand(*train_X.shape)), axis=0)
# train_y = np.concatenate((train_y, train_y, train_y), axis=0)

test_X = data['Index'][:].values.reshape(-1, 1)

gpr = GPR(optimize=True)
gpr.fit(train_X, train_y, alpha_range=(0.5,2.5), r_range=(1e2, 1e3), v2_range=(1e-4,1))
mu, std = gpr.predict(test_X)

with visualization(axsize=(3,1)) as (vis, fig, axes):
    vis.set_data(test_X.reshape(-1), mu.reshape(-1), std)
    vis.plot_predict_result(axes[0])
    vis.plot_confidence_interval(axes[0], 0.95)
    axes[0].plot(data['Close'], label='Close')
    axes[0].set_title(", ".join([param + "=" + f"{gpr.params[param]:.2f}" for param in gpr.params]))
    axes[0].grid('on')