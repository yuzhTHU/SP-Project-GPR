import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GPR import GPR
from visualization import visualization

## Data set load
pd.set_option('display.notebook_repr_html',False)
data = pd.read_excel(r'grid.xls')[:-1]
data.insert(loc=0, column='Index', value=np.arange(data.shape[0]))

with visualization(figsize=(8,5)) as (vis, fig, ax):
    ax.set_title('Data in grid.xls')
    ax.plot(data['Close'], label='Close', zorder=10)
    ax.plot(data['Open'], label='Open', alpha=0.3)
    ax.plot(data['High'], label='High', alpha=0.3)
    ax.plot(data['Low'], label='Low', alpha=0.3)
    ax.grid('on')

print(np.sum(np.diff(data['Close'].values)>0) / len(np.diff(data['Close'])))

## Visualization of Different Kernel Functions
gpr = GPR(optimize=False)
with visualization(axsize=(4,1), figsize=(5,8), show=False) as (vis, fig, axes):
    vis.plot_kernal(axes[0], gpr.squared_exponential_kernel, name="Squared Exponential Kernel")
    vis.plot_kernal(axes[1], gpr.LS_squared_exponential_kernel, name="LS Squared Exponential Kernel")
    vis.plot_kernal(axes[2], gpr.sin_exponential_kernel, name="Sin Exponential Kernel")
    vis.plot_kernal(axes[3], gpr.decay_sin_exponential_kernel, name="Decay Sin Exponential Kernel")
    plt.suptitle('Visualization of Different Kernel Functions')
    plt.subplots_adjust(left=.106,bottom=.035,right=.963,top=.902,wspace=.2,hspace=.389)

## Short Term Predict
seq_len = 6
pred_len = 1
mu = np.zeros(len(data['Close']))
mu[:seq_len] = data['Close'][:seq_len]
std = np.zeros(len(data['Close']))
gpr = GPR(optimize=True, kernel='decay_sin_exp')
gpr.fit(data['Index'].values.reshape(-1, 1), data['Close'].values.reshape(-1, 1), alpha=(1.5,2.5), r=(1e-4,1e4), v0=(1e-4,1e4), v2=(1e-4,1))
gpr.optimize=False
with visualization(show=False) as (vis, fig, ax):
    for i in range(seq_len, len(data['Close']), pred_len):
        gpr.fit(
            data['Index'].values[i - seq_len:i].reshape(-1, 1), 
            data['Close'].values[i - seq_len:i].reshape(-1, 1),
            alpha=(1.5,2.5), r=(1e-4,1e4), v0=(1e-4,1e4), v2=(1e-4,1)
        )
        mmu, sstd = gpr.predict(data['Index'].values[i:i + pred_len].reshape(-1, 1))
        mu[i:i + pred_len] = mmu.reshape(-1)
        std[i:i + pred_len] = sstd
        ax.cla()
        ax.plot(data['Index'][i - seq_len:i+pred_len].values.reshape(-1, 1), data['Close'][i - seq_len:i+pred_len].values.reshape(-1, 1), label='Real')
        vis.set_data(data['Index'][i:i + pred_len].values.reshape(-1), mmu.reshape(-1), sstd)
        vis.plot_predict_result(ax)
        vis.plot_confidence_interval(ax, 0.95)
        ax.set_title(f'index={i}')
        # plt.pause(0.05)

with visualization(axsize=(1, 1), figsize=(8, 6)) as (vis, fig, ax):
    vis.set_data(data['Index'].values.reshape(-1), mu, std)
    vis.plot_confidence_interval(ax, 0.95)
    vis.plot_predict_result(ax)
    ax.plot(data['Close'][:], label='Real')
    ax.grid('on')
    accuracy = np.sum(np.diff(mu[seq_len:]) * np.diff(data['Close'][seq_len:].values) >= 0) / len(mu[seq_len:])
    ax.set_title(f'Decay Sin Exponential Kernel (Predict {pred_len} day from {seq_len} previous day), ACCURACY={accuracy*100:.2f}%')

## Long Term Predict
train_X = data['Index'][:200:1].values.reshape(-1, 1)
train_y = data['Close'][:200:1].values.reshape(-1, 1)
test_X = np.linspace(data['Index'][0], data['Index'].values[-1], 1000).reshape(-1, 1)
gpr = GPR(optimize=True, kernel='squared_exp')
gpr.fit(train_X, train_y, r=(1e-4,1e4), v0=(1e-4,1e4), v2=(1e-4,1e4), alpha=(1.0, 3.0))
mu, std = gpr.predict(test_X)

with visualization(axsize=(2,1), figsize=(8,6)) as (vis, fig, axes):
    vis.set_data(test_X.reshape(-1), mu.reshape(-1), std)
    vis.plot_predict_result(axes[0])
    vis.plot_confidence_interval(axes[0], 0.95)
    axes[0].plot(data['Close'][:], label='Real')
    # axes[0].scatter(data['Index'][:11], data['Close'][:11], marker='x')
    plt.suptitle("Squared Exponential Kernel")
    axes[0].grid('on')
    # vis.show_param(axes[0], gpr)
    vis.plot_kernal(axes[1], gpr.kernel)

## Interpolation
train_X = data['Index'][:200:20].values.reshape(-1, 1)
train_y = data['Close'][:200:20].values.reshape(-1, 1)
test_X = np.linspace(data['Index'][0], data['Index'].values[-1], 1000).reshape(-1, 1)
with visualization(axsize=(2,1), figsize=(8,6)) as (vis, fig, axes):
    gpr = GPR(optimize=True, kernel='squared_exp')
    gpr.fit(train_X, train_y, r=(1e2,1e4), v0=(1e-4,1e4), v2=(1e-4,1e4), alpha=(1.0, 3.0))
    mu, std = gpr.predict(test_X)
    vis.set_data(test_X.reshape(-1), mu.reshape(-1), std)
    vis.plot_predict_result(axes[0])
    vis.plot_confidence_interval(axes[0], 0.95)
    axes[0].plot(data['Close'][:], label='Real')
    axes[0].scatter(train_X, train_y, color='red', marker='o', label='train data', zorder=10)
    axes[0].set_title('With Noise Item in Kernel')
    axes[0].grid('on')
    axes[0].legend()

    gpr = GPR(optimize=True, kernel='squared_exp')
    gpr.fit(train_X, train_y, r=(1e2,1e4), v0=(1e-4,1e4), v2=0, alpha=(1.0, 3.0))
    mu, std = gpr.predict(test_X)
    vis.set_data(test_X.reshape(-1), mu.reshape(-1), std)
    vis.plot_predict_result(axes[1])
    vis.plot_confidence_interval(axes[1], 0.95)
    axes[1].plot(data['Close'][:], label='Real')
    axes[1].scatter(train_X, train_y, color='red', marker='o', label='train data', zorder=10)
    axes[1].set_title('Without Noise Item in Kernel')
    axes[1].grid('on')
    axes[1].legend()

    fig.suptitle('Interpolation with Squared Exp Kernel')


## 2-D Period Predict
data = pd.read_csv(f"CELYAD ONCOLOGY SA (06-19-2015 _ 01-14-2022).csv")
data.insert(loc=0, column='Index', value=np.arange(data.shape[0]))
period = 365

with visualization(figsize=(8,5)) as (vis, fig, ax):
    ax.set_title('Data in CELYAD ONCOLOGY SA (06-19-2015 _ 01-14-2022).csv')
    ax.plot(data['Close'], label='Close', zorder=10)
    ax.plot(data['Open'], label='Open', alpha=0.3)
    ax.plot(data['High'], label='High', alpha=0.3)
    ax.plot(data['Low'], label='Low', alpha=0.3)
    ax.grid('on')

all_data = data['Close'].values
all_data = all_data[:len(all_data) // period * period].reshape([-1, period])
all_data = ((all_data - np.mean(all_data, axis=1, keepdims=True)) / np.std(all_data, axis=1, keepdims=True)).reshape([-1, 1])
all_index = np.stack((np.arange(len(all_data)) // period, np.arange(len(all_data)) % period), axis=1)
train_data = all_data[:-180, :]
train_index = all_index[:-180, :]
test_data = all_data[-period:, :]
test_index = all_index[-period:, :]

gpr = GPR(optimize=False, kernel='decay_sin_exp')
gpr.params['r'] = [2.0, 2.0]
gpr.params['r2'] = [20.0, 20.0]
gpr.fit(train_index, train_data, alpha=2.2, v0=8, v2=1e-1, v02=5, nu=0.15)
mu, std = gpr.predict(test_index)
mu = mu.reshape(-1)
with visualization(axsize=(1,1)) as (vis, fig, ax):
    vis.set_data(test_index[:, 1], mu, std)
    vis.plot_predict_result(ax)
    vis.plot_confidence_interval(ax, 0.95)
    ax.plot(test_index[:, 1], test_data, label='Real')
    ax.grid('on')
    ax.legend()
    plt.suptitle("Decay Sin Exponential Kernel")

# # Method 2
# train_index1 = np.repeat(np.arange(period).reshape(1,-1), train_data.shape[0], axis=0)
# train_index2 = np.repeat(np.arange(period).reshape(1,-1), train_data.shape[0], axis=0)
# test_index = train_index[0, :]
# gpr = GPR(optimize=True, kernel='squared_exp')
# gpr.fit(train_index.reshape(-1, 1), train_data.reshape(-1, 1), 
#     alpha=(1.5,3.5), r=(1e-4,1e4), v0=(1e-4,1e4), v2=(1e-4,1))
# mu, std = gpr.predict(test_index.reshape(-1, 1))
# mu = mu.reshape(-1)

# with visualization(axsize=(2,1)) as (vis, fig, axes):
#     vis.set_data(test_index[:, 1], mu, std)
#     vis.plot_predict_result(axes[0])
#     vis.plot_confidence_interval(axes[0], 0.95)
#     axes[0].plot(test_index[:, 1], test_data, label='Real')
#     for i in range(train_data.shape[0]):
#         axes[1].plot(train_index[i], train_data[i])
#     axes[0].grid('on')
#     axes[1].grid('on')
#     axes[0].legend()
#     plt.suptitle("Squared Exponential Kernel")