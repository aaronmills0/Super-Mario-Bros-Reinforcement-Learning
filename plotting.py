import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

fig_x = 8
fig_y = 8


ppo5000 = pd.read_csv("data/ppo_rewards_5000.csv")
ppo10000 = pd.read_csv("data/ppo_rewards_10000.csv")
ppo15000 = pd.read_csv("data/ppo_rewards_15000.csv")
ppo20000 = pd.read_csv("data/ppo_rewards_20000.csv")
ppo25000 = pd.read_csv("data/ppo_rewards_25000.csv")
ppo30000 = pd.read_csv("data/ppo_rewards_30000.csv")
ppo35000 = pd.read_csv("data/ppo_rewards_35000.csv")
ppo40000 = pd.read_csv("data/ppo_rewards_40000.csv")
ppo45000 = pd.read_csv("data/ppo_rewards_45000.csv")
ppo50000 = pd.read_csv("data/ppo_rewards_50000.csv")
ppo55000 = pd.read_csv("data/ppo_rewards_55000.csv")
ppo60000 = pd.read_csv("data/ppo_rewards_60000.csv")

del ppo5000[ppo5000.columns[0]]
del ppo10000[ppo10000.columns[0]]
del ppo15000[ppo15000.columns[0]]
del ppo20000[ppo20000.columns[0]]
del ppo25000[ppo25000.columns[0]]
del ppo30000[ppo30000.columns[0]]
del ppo35000[ppo35000.columns[0]]
del ppo40000[ppo40000.columns[0]]
del ppo45000[ppo45000.columns[0]]
del ppo50000[ppo50000.columns[0]]
del ppo55000[ppo55000.columns[0]]
del ppo60000[ppo60000.columns[0]]

ppo5 = ppo5000.sum(axis=1)
ppo10 = ppo10000.sum(axis=1)
ppo15 = ppo15000.sum(axis=1)
ppo20 = ppo20000.sum(axis=1)
ppo25 = ppo25000.sum(axis=1)
ppo30 = ppo30000.sum(axis=1)
ppo35 = ppo35000.sum(axis=1)
ppo40 = ppo40000.sum(axis=1)
ppo45 = ppo45000.sum(axis=1)
ppo50 = ppo50000.sum(axis=1)
ppo55 = ppo55000.sum(axis=1)
ppo60 = ppo60000.sum(axis=1)

ppo_total = pd.concat([ppo5, ppo10, ppo15, ppo20, ppo25, ppo30, ppo35, ppo40, ppo45, ppo50, ppo55, ppo60])

dueling5000 = pd.read_csv("data/dueling_dqn_rewards_5000.csv")
dueling10000 = pd.read_csv("data/dueling_dqn_rewards_10000.csv")
dueling15000 = pd.read_csv("data/dueling_dqn_rewards_15000.csv")
dueling20000 = pd.read_csv("data/dueling_dqn_rewards_20000.csv")
dueling25000 = pd.read_csv("data/dueling_dqn_rewards_25000.csv")
dueling30000 = pd.read_csv("data/dueling_dqn_rewards_30000.csv")
dueling35000 = pd.read_csv("data/dueling_dqn_rewards_35000.csv")
dueling40000 = pd.read_csv("data/dueling_dqn_rewards_40000.csv")
dueling45000 = pd.read_csv("data/dueling_dqn_rewards_45000.csv")
dueling50000 = pd.read_csv("data/dueling_dqn_rewards_50000.csv")
dueling55000 = pd.read_csv("data/dueling_dqn_rewards_55000.csv")
dueling60000 = pd.read_csv("data/dueling_dqn_rewards_60000.csv")

del dueling5000[dueling5000.columns[0]]
del dueling10000[dueling10000.columns[0]]
del dueling15000[dueling15000.columns[0]]
del dueling20000[dueling20000.columns[0]]
del dueling25000[dueling25000.columns[0]]
del dueling30000[dueling30000.columns[0]]
del dueling35000[dueling35000.columns[0]]
del dueling40000[dueling40000.columns[0]]
del dueling45000[dueling45000.columns[0]]
del dueling50000[dueling50000.columns[0]]
del dueling55000[dueling55000.columns[0]]
del dueling60000[dueling60000.columns[0]]

dueling5 = dueling5000.sum(axis=1)
dueling10 = dueling10000.sum(axis=1)
dueling15 = dueling15000.sum(axis=1)
dueling20 = dueling20000.sum(axis=1)
dueling25 = dueling25000.sum(axis=1)
dueling30 = dueling30000.sum(axis=1)
dueling35 = dueling35000.sum(axis=1)
dueling40 = dueling40000.sum(axis=1)
dueling45 = dueling45000.sum(axis=1)
dueling50 = dueling50000.sum(axis=1)
dueling55 = dueling55000.sum(axis=1)
dueling60 = dueling60000.sum(axis=1)

dueling_total = pd.concat([dueling5, dueling10, dueling15, dueling20, dueling25, dueling30, dueling35, dueling40, dueling45, dueling50, dueling55, dueling60])

ppo_wins5000 = pd.read_csv("data/ppo_wins_5000.csv")
ppo_wins10000 = pd.read_csv("data/ppo_wins_10000.csv")
ppo_wins15000 = pd.read_csv("data/ppo_wins_15000.csv")
ppo_wins20000 = pd.read_csv("data/ppo_wins_20000.csv")
ppo_wins25000 = pd.read_csv("data/ppo_wins_25000.csv")
ppo_wins30000 = pd.read_csv("data/ppo_wins_30000.csv")
ppo_wins35000 = pd.read_csv("data/ppo_wins_35000.csv")
ppo_wins40000 = pd.read_csv("data/ppo_wins_40000.csv")
ppo_wins45000 = pd.read_csv("data/ppo_wins_45000.csv")
ppo_wins50000 = pd.read_csv("data/ppo_wins_50000.csv")
ppo_wins55000 = pd.read_csv("data/ppo_wins_55000.csv")
ppo_wins60000 = pd.read_csv("data/ppo_wins_60000.csv")

del ppo_wins5000[ppo_wins5000.columns[0]]
del ppo_wins10000[ppo_wins10000.columns[0]]
del ppo_wins15000[ppo_wins15000.columns[0]]
del ppo_wins20000[ppo_wins20000.columns[0]]
del ppo_wins25000[ppo_wins25000.columns[0]]
del ppo_wins30000[ppo_wins30000.columns[0]]
del ppo_wins35000[ppo_wins35000.columns[0]]
del ppo_wins40000[ppo_wins40000.columns[0]]
del ppo_wins45000[ppo_wins45000.columns[0]]
del ppo_wins50000[ppo_wins50000.columns[0]]
del ppo_wins55000[ppo_wins55000.columns[0]]
del ppo_wins60000[ppo_wins60000.columns[0]]

ppo_wins5000 = ppo_wins5000*1
ppo_wins10000 = ppo_wins10000*1
ppo_wins15000 = ppo_wins15000*1
ppo_wins20000 = ppo_wins20000*1
ppo_wins25000 = ppo_wins25000*1
ppo_wins30000 = ppo_wins30000*1
ppo_wins35000 = ppo_wins35000*1
ppo_wins40000 = ppo_wins40000*1
ppo_wins45000 = ppo_wins45000*1
ppo_wins50000 = ppo_wins50000*1
ppo_wins55000 = ppo_wins55000*1
ppo_wins60000 = ppo_wins60000*1

ppo_wins5 = ppo_wins5000.sum(axis=1)
ppo_wins10 = ppo_wins10000.sum(axis=1)
ppo_wins15 = ppo_wins15000.sum(axis=1)
ppo_wins20 = ppo_wins20000.sum(axis=1)
ppo_wins25 = ppo_wins25000.sum(axis=1)
ppo_wins30 = ppo_wins30000.sum(axis=1)
ppo_wins35 = ppo_wins35000.sum(axis=1)
ppo_wins40 = ppo_wins40000.sum(axis=1)
ppo_wins45 = ppo_wins45000.sum(axis=1)
ppo_wins50 = ppo_wins50000.sum(axis=1)
ppo_wins55 = ppo_wins55000.sum(axis=1)
ppo_wins60 = ppo_wins60000.sum(axis=1)

ppo_wins_total = pd.concat([ppo_wins5, ppo_wins10, ppo_wins15, ppo_wins20, ppo_wins25, ppo_wins30, ppo_wins35, ppo_wins40, ppo_wins45, ppo_wins50, ppo_wins55, ppo_wins60])

window_size = 5000
ppo_wins_total = ppo_wins_total.rolling(min_periods=0, window=window_size).sum()

dueling_wins5000 = pd.read_csv("data/dueling_dqn_wins_5000.csv")
dueling_wins10000 = pd.read_csv("data/dueling_dqn_wins_10000.csv")
dueling_wins15000 = pd.read_csv("data/dueling_dqn_wins_15000.csv")
dueling_wins20000 = pd.read_csv("data/dueling_dqn_wins_20000.csv")
dueling_wins25000 = pd.read_csv("data/dueling_dqn_wins_25000.csv")
dueling_wins30000 = pd.read_csv("data/dueling_dqn_wins_30000.csv")
dueling_wins35000 = pd.read_csv("data/dueling_dqn_wins_35000.csv")
dueling_wins40000 = pd.read_csv("data/dueling_dqn_wins_40000.csv")
dueling_wins45000 = pd.read_csv("data/dueling_dqn_wins_45000.csv")
dueling_wins50000 = pd.read_csv("data/dueling_dqn_wins_50000.csv")
dueling_wins55000 = pd.read_csv("data/dueling_dqn_wins_55000.csv")
dueling_wins60000 = pd.read_csv("data/dueling_dqn_wins_60000.csv")

del dueling_wins5000[dueling_wins5000.columns[0]]
del dueling_wins10000[dueling_wins10000.columns[0]]
del dueling_wins15000[dueling_wins15000.columns[0]]
del dueling_wins20000[dueling_wins20000.columns[0]]
del dueling_wins25000[dueling_wins25000.columns[0]]
del dueling_wins30000[dueling_wins30000.columns[0]]
del dueling_wins35000[dueling_wins35000.columns[0]]
del dueling_wins40000[dueling_wins40000.columns[0]]
del dueling_wins45000[dueling_wins45000.columns[0]]
del dueling_wins50000[dueling_wins50000.columns[0]]
del dueling_wins55000[dueling_wins55000.columns[0]]
del dueling_wins60000[dueling_wins60000.columns[0]]

dueling_wins5000 = dueling_wins5000*1
dueling_wins10000 = dueling_wins10000*1
dueling_wins15000 = dueling_wins15000*1
dueling_wins20000 = dueling_wins20000*1
dueling_wins25000 = dueling_wins25000*1
dueling_wins30000 = dueling_wins30000*1
dueling_wins35000 = dueling_wins35000*1
dueling_wins40000 = dueling_wins40000*1
dueling_wins45000 = dueling_wins45000*1
dueling_wins50000 = dueling_wins50000*1
dueling_wins55000 = dueling_wins55000*1
dueling_wins60000 = dueling_wins60000*1

dueling_wins5 = dueling_wins5000.sum(axis=1)
dueling_wins10 = dueling_wins10000.sum(axis=1)
dueling_wins15 = dueling_wins15000.sum(axis=1)
dueling_wins20 = dueling_wins20000.sum(axis=1)
dueling_wins25 = dueling_wins25000.sum(axis=1)
dueling_wins30 = dueling_wins30000.sum(axis=1)
dueling_wins35 = dueling_wins35000.sum(axis=1)
dueling_wins40 = dueling_wins40000.sum(axis=1)
dueling_wins45 = dueling_wins45000.sum(axis=1)
dueling_wins50 = dueling_wins50000.sum(axis=1)
dueling_wins55 = dueling_wins55000.sum(axis=1)
dueling_wins60 = dueling_wins60000.sum(axis=1)

dueling_wins_total = pd.concat([dueling_wins5, dueling_wins10, dueling_wins15, dueling_wins20, dueling_wins25, dueling_wins30, dueling_wins35, dueling_wins40, dueling_wins45, dueling_wins50, dueling_wins55, dueling_wins60])

window_size = 5000
dueling_wins_total = dueling_wins_total.rolling(min_periods=0, window=window_size).sum()

x = list(range(1, 60001))

PPO_Spline = make_interp_spline(x, ppo_total)
PPO_WINS_Spline = make_interp_spline(x, ppo_wins_total)

DUELING_Spline = make_interp_spline(x, dueling_total)
DUELING_WINS_Spline = make_interp_spline(x, dueling_wins_total)

X_ = np.linspace(min(x), max(x), 300)

X_WINS = np.linspace(min(x), max(x), 500)
X_TOTAL = np.linspace(min(x), max(x), 60000)

PPO_Y = PPO_Spline(X_)
PPO_WINS_Y = PPO_WINS_Spline(X_WINS)

DUELING_Y = DUELING_Spline(X_)
DUELING_WINS_Y = DUELING_WINS_Spline(X_WINS)

PPO_Y_TOTAL = PPO_Spline(X_TOTAL)

DUELING_Y_TOTAL = DUELING_Spline(X_TOTAL)

plt.figure(figsize=(fig_x, fig_y))
plt.rcParams.update({'font.size': 18})
plt.title("Return (Smoothed)")
plt.ylabel("Return")
plt.xlabel("Episode")
plt.plot(x, DUELING_Y_TOTAL, color='lightgreen', alpha=0.5, label='Dueling DQN Total')

plt.plot(x, PPO_Y_TOTAL, color='lightblue', alpha=0.5, label='PPO Total')

plt.plot(X_, DUELING_Y, color='green', label='Dueling DQN')

plt.plot(X_, PPO_Y, color='blue', label='PPO')

plt.legend(loc=1)
plt.savefig('figures/reward.pdf')
plt.show()


plt.figure(figsize=(fig_x, fig_y))

plt.title(f"Wins Over Previous {window_size} Episodes (Smoothed)")
plt.ylabel("Wins")
plt.xlabel("Episode")
plt.plot(X_WINS, DUELING_WINS_Y, color='green', label='Dueling DQN')

plt.plot(X_WINS, PPO_WINS_Y, color='blue', label='PPO')

plt.legend(loc=1)
plt.savefig('figures/wins.pdf')
plt.show()

