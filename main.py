import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from demo import GPR

pd.set_option('display.notebook_repr_html',False)
data = pd.read_excel(r'grid.xls')[:-1]

data.insert(loc=0, column='Index', value=np.arange(data.shape[0]))
print(data)

gpr = GPR()
gpr.fit(data['Index'][:200], data['Close'][:200])
mu, std = gpr.predict(data['Index'])



fig, ax = plt.subplots(1, 1)
ax.plot(data['Open'], label='Open')
ax.plot(data['High'], label='High')
ax.plot(data['Low'], label='Low')
ax.plot(data['Close'], label='Close')
# ax.plot(data['Volume'], label='Volume')
ax.legend()
plt.show()
