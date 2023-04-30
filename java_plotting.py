import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

fig_x = 8
fig_y = 8

x = list(range(1, 50001))

X_ = np.linspace(min(x), max(x), 300)

X_TOTAL = np.linspace(min(x), max(x), 50000)

dql1 = pd.read_csv("java_data/doubleqlearning_run1.csv", header=None)
dql2 = pd.read_csv("java_data/doubleqlearning_run2.csv", header=None)
dql3 = pd.read_csv("java_data/doubleqlearning_run3.csv", header=None)
dql4 = pd.read_csv("java_data/doubleqlearning_run4.csv", header=None)
dql5 = pd.read_csv("java_data/doubleqlearning_run5.csv", header=None)

dql1_distance = dql1.iloc[:, 0]
dql1_reward = dql1.iloc[:, 2]
dql2_distance = dql2.iloc[:, 0]
dql2_reward = dql2.iloc[:, 2]
dql3_distance = dql3.iloc[:, 0]
dql3_reward = dql3.iloc[:, 2]
dql4_distance = dql4.iloc[:, 0]
dql4_reward = dql4.iloc[:, 2]
dql5_distance = dql5.iloc[:, 0]
dql5_reward = dql5.iloc[:, 2]

dql_distance = pd.concat([dql1_distance, dql2_distance, dql3_distance, dql4_distance, dql5_distance], axis=1)
dql_reward = pd.concat([dql1_reward, dql2_reward, dql3_reward, dql4_reward, dql5_reward], axis=1)

dql_distance_mean = dql_distance.mean(axis=1)
dql_distance_std = dql_distance.std(axis=1)

dql_reward_mean = dql_reward.mean(axis=1)
dql_reward_std = dql_reward.std(axis=1)

DQL_DISTANCE_MEAN_Spline = make_interp_spline(x, dql_distance_mean)
DQL_DISTANCE_STD_Spline = make_interp_spline(x, dql_distance_std)

DQL_REWARD_MEAN_Spline = make_interp_spline(x, dql_reward_mean)
DQL_REWARD_STD_Spline = make_interp_spline(x, dql_reward_std)

dql_distance_mean_y = DQL_DISTANCE_MEAN_Spline(X_)
dql_distance_std_y = DQL_DISTANCE_STD_Spline(X_)

dql_reward_mean_y = DQL_REWARD_MEAN_Spline(X_)
dql_reward_std_y = DQL_REWARD_STD_Spline(X_)

ql1 = pd.read_csv("java_data/qlearning_run1.csv", header=None)
ql2 = pd.read_csv("java_data/qlearning_run2.csv", header=None)
ql3 = pd.read_csv("java_data/qlearning_run3.csv", header=None)
ql4 = pd.read_csv("java_data/qlearning_run4.csv", header=None)
ql5 = pd.read_csv("java_data/qlearning_run5.csv", header=None)

ql1_distance = ql1.iloc[:, 0]
ql1_reward = ql1.iloc[:, 2]
ql2_distance = ql2.iloc[:, 0]
ql2_reward = ql2.iloc[:, 2]
ql3_distance = ql3.iloc[:, 0]
ql3_reward = ql3.iloc[:, 2]
ql4_distance = ql4.iloc[:, 0]
ql4_reward = ql4.iloc[:, 2]
ql5_distance = ql5.iloc[:, 0]
ql5_reward = ql5.iloc[:, 2]

ql_distance = pd.concat([ql1_distance, ql2_distance, ql3_distance, ql4_distance, ql5_distance], axis=1)
ql_reward = pd.concat([ql1_reward, ql2_reward, ql3_reward, ql4_reward, ql5_reward], axis=1)

ql_distance_mean = ql_distance.mean(axis=1)
ql_distance_std = ql_distance.std(axis=1)

ql_reward_mean = ql_reward.mean(axis=1)
ql_reward_std = ql_reward.std(axis=1)

QL_DISTANCE_MEAN_Spline = make_interp_spline(x, ql_distance_mean)
QL_DISTANCE_STD_Spline = make_interp_spline(x, ql_distance_std)

QL_REWARD_MEAN_Spline = make_interp_spline(x, ql_reward_mean)
QL_REWARD_STD_Spline = make_interp_spline(x, ql_reward_std)

ql_distance_mean_y = QL_DISTANCE_MEAN_Spline(X_)
ql_distance_std_y = QL_DISTANCE_STD_Spline(X_)

ql_reward_mean_y = QL_REWARD_MEAN_Spline(X_)
ql_reward_std_y = QL_REWARD_STD_Spline(X_)

sarsa1 = pd.read_csv("java_data/sarsa_run1.csv", header=None)
sarsa2 = pd.read_csv("java_data/sarsa_run2.csv", header=None)
sarsa3 = pd.read_csv("java_data/sarsa_run3.csv", header=None)
sarsa4 = pd.read_csv("java_data/sarsa_run4.csv", header=None)
sarsa5 = pd.read_csv("java_data/sarsa_run5.csv", header=None)

sarsa1_distance = sarsa1.iloc[:, 0]
sarsa1_reward = sarsa1.iloc[:, 2]
sarsa2_distance = sarsa2.iloc[:, 0]
sarsa2_reward = sarsa2.iloc[:, 2]
sarsa3_distance = sarsa3.iloc[:, 0]
sarsa3_reward = sarsa3.iloc[:, 2]
sarsa4_distance = sarsa4.iloc[:, 0]
sarsa4_reward = sarsa4.iloc[:, 2]
sarsa5_distance = sarsa5.iloc[:, 0]
sarsa5_reward = sarsa5.iloc[:, 2]

sarsa_distance = pd.concat([sarsa1_distance, sarsa2_distance, sarsa3_distance, sarsa4_distance, sarsa5_distance], axis=1)
sarsa_reward = pd.concat([sarsa1_reward, sarsa2_reward, sarsa3_reward, sarsa4_reward, sarsa5_reward], axis=1)

sarsa_distance_mean = sarsa_distance.mean(axis=1)
sarsa_distance_std = sarsa_distance.std(axis=1)

sarsa_reward_mean = sarsa_reward.mean(axis=1)
sarsa_reward_std = sarsa_reward.std(axis=1)

SARSA_DISTANCE_MEAN_Spline = make_interp_spline(x, sarsa_distance_mean)
SARSA_DISTANCE_STD_Spline = make_interp_spline(x, sarsa_distance_std)

SARSA_REWARD_MEAN_Spline = make_interp_spline(x, sarsa_reward_mean)
SARSA_REWARD_STD_Spline = make_interp_spline(x, sarsa_reward_std)

sarsa_distance_mean_y = SARSA_DISTANCE_MEAN_Spline(X_)
sarsa_distance_std_y = SARSA_DISTANCE_STD_Spline(X_)

sarsa_reward_mean_y = SARSA_REWARD_MEAN_Spline(X_)
sarsa_reward_std_y = SARSA_REWARD_STD_Spline(X_)

expectedsarsa1 = pd.read_csv("java_data/expectedsarsa_run1.csv", header=None)
expectedsarsa2 = pd.read_csv("java_data/expectedsarsa_run2.csv", header=None)
expectedsarsa3 = pd.read_csv("java_data/expectedsarsa_run3.csv", header=None)
expectedsarsa4 = pd.read_csv("java_data/expectedsarsa_run4.csv", header=None)
expectedsarsa5 = pd.read_csv("java_data/expectedsarsa_run5.csv", header=None)

expectedsarsa1_distance = expectedsarsa1.iloc[:, 0]
expectedsarsa1_reward = expectedsarsa1.iloc[:, 2]
expectedsarsa2_distance = expectedsarsa2.iloc[:, 0]
expectedsarsa2_reward = expectedsarsa2.iloc[:, 2]
expectedsarsa3_distance = expectedsarsa3.iloc[:, 0]
expectedsarsa3_reward = expectedsarsa3.iloc[:, 2]
expectedsarsa4_distance = expectedsarsa4.iloc[:, 0]
expectedsarsa4_reward = expectedsarsa4.iloc[:, 2]
expectedsarsa5_distance = expectedsarsa5.iloc[:, 0]
expectedsarsa5_reward = expectedsarsa5.iloc[:, 2]

expectedsarsa_distance = pd.concat([expectedsarsa1_distance, expectedsarsa2_distance, expectedsarsa3_distance, expectedsarsa4_distance, expectedsarsa5_distance], axis=1)
expectedsarsa_reward = pd.concat([expectedsarsa1_reward, expectedsarsa2_reward, expectedsarsa3_reward, expectedsarsa4_reward, expectedsarsa5_reward], axis=1)

expectedsarsa_distance_mean = expectedsarsa_distance.mean(axis=1)
expectedsarsa_distance_std = expectedsarsa_distance.std(axis=1)

expectedsarsa_reward_mean = expectedsarsa_reward.mean(axis=1)
expectedsarsa_reward_std = expectedsarsa_reward.std(axis=1)

EXPECTEDSARSA_DISTANCE_MEAN_Spline = make_interp_spline(x, expectedsarsa_distance_mean)
EXPECTEDSARSA_DISTANCE_STD_Spline = make_interp_spline(x, expectedsarsa_distance_std)

EXPECTEDSARSA_REWARD_MEAN_Spline = make_interp_spline(x, expectedsarsa_reward_mean)
EXPECTEDSARSA_REWARD_STD_Spline = make_interp_spline(x, expectedsarsa_reward_std)

expectedsarsa_distance_mean_y = EXPECTEDSARSA_DISTANCE_MEAN_Spline(X_)
expectedsarsa_distance_std_y = EXPECTEDSARSA_DISTANCE_STD_Spline(X_)

expectedsarsa_reward_mean_y = EXPECTEDSARSA_REWARD_MEAN_Spline(X_)
expectedsarsa_reward_std_y = EXPECTEDSARSA_REWARD_STD_Spline(X_)
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(fig_x, fig_y))
plt.title("Mean Return (Smoothed)")
plt.ylabel("Mean Return")
plt.xlabel("Episode")
plt.plot(X_, ql_reward_mean_y, color='green', label='Q-Learning')
plt.fill_between(X_, ql_reward_mean_y-ql_reward_std_y, ql_reward_mean_y+ql_reward_std_y, color='lightgreen', alpha=0.3)
plt.plot(X_, sarsa_reward_mean_y, color='orange', label='SARSA')
plt.plot(X_, dql_reward_mean_y, color='blue', label='Double Q-Learning')
plt.fill_between(X_, dql_reward_mean_y-dql_reward_std_y, dql_reward_mean_y+dql_reward_std_y, color='lightblue', alpha=0.3)
plt.fill_between(X_, sarsa_reward_mean_y-sarsa_reward_std_y, sarsa_reward_mean_y+sarsa_reward_std_y, color='bisque', alpha=0.3)
plt.plot(X_, expectedsarsa_reward_mean_y, color='firebrick', label='Expected SARSA')
plt.fill_between(X_, expectedsarsa_reward_mean_y-expectedsarsa_reward_std_y, expectedsarsa_reward_mean_y+expectedsarsa_reward_std_y, color='lightcoral', alpha=0.3)
plt.legend()
plt.savefig('figures/meanreturn.pdf')
plt.show()

plt.figure(figsize=(fig_x, fig_y))
plt.title("Mean Fraction Completed (Smoothed)")
plt.ylabel("Fraction Completed")
plt.xlabel("Episode")
plt.plot(X_, ql_distance_mean_y, color='green', label='Q-Learning')
plt.fill_between(X_, ql_distance_mean_y-ql_distance_std_y, ql_distance_mean_y+ql_distance_std_y, color='lightgreen', alpha=0.3)
plt.plot(X_, sarsa_distance_mean_y, color='orange', label='SARSA')
plt.fill_between(X_, sarsa_distance_mean_y-sarsa_distance_std_y, sarsa_distance_mean_y+sarsa_distance_std_y, color='bisque', alpha=0.3)
plt.plot(X_, dql_distance_mean_y, color='blue', label='Double Q-Learning')
plt.fill_between(X_, dql_distance_mean_y-dql_distance_std_y, dql_distance_mean_y+dql_distance_std_y, color='lightblue', alpha=0.3)
plt.plot(X_, expectedsarsa_distance_mean_y, color='firebrick', label='Expected SARSA')
plt.fill_between(X_, expectedsarsa_distance_mean_y-expectedsarsa_distance_std_y, expectedsarsa_distance_mean_y+expectedsarsa_distance_std_y, color='lightcoral', alpha=0.3)
plt.ylim(0, 1.0)
plt.legend()
plt.savefig('figures/meandistance.pdf')
plt.show()

