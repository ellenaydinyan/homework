"""
A/B testing of four advertisement options with Epsilon-Greedy and Thompson Sampling.

Bandit_Reward = [1, 2, 3, 4], NumberOfTrials = 20000.
Each ad (arm) gives a Gaussian reward with mean equal to its Bandit_Reward value
and known precision tau = 1 (standard deviation 1).

Outputs:
    - rewards.csv                      : {Bandit, Reward, Algorithm} for every trial
    - plot1_epsilon_greedy.png         : learning process of Epsilon-Greedy (linear and log)
    - plot1_thompson_sampling.png      : learning process of Thompson Sampling (linear and log)
    - plot2_comparison.png             : cumulative rewards and regrets of both algorithms
"""
############################### LOGGER
from abc import ABC, abstractmethod
import os

import matplotlib

matplotlib.use("Agg")  # save plots to files, works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
EPSILON = 1.0          # starting epsilon, decayed as epsilon / t
PRECISION = 1.0        # known precision (1 / variance) of the reward of each arm
SEED = 42
CSV_FILE = "rewards.csv"


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """Plots for the learning process and the comparison of the algorithms."""

    def plot1(self, bandit, file_name=None):
        """
        Visualize the learning process of one algorithm on a linear and a log scale.

        The left plots show the running average reward against the best possible
        mean reward. The right plots show how often each arm was chosen.

        :param bandit: a finished EpsilonGreedy or ThompsonSampling object
        :param file_name: where to save the figure (default: plot1_<algorithm>.png)
        """
        rewards = np.array(bandit.rewards)
        trials = np.arange(1, len(rewards) + 1)
        running_avg = np.cumsum(rewards) / trials
        best = max(bandit.p)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, scale in zip(axes, ["linear", "log"]):
            ax.plot(trials, running_avg, label="Running average reward")
            ax.axhline(best, color="red", linestyle="--", label=f"Optimal mean = {best}")
            ax.set_xscale(scale)
            ax.set_title(f"{bandit.name}: learning process ({scale} scale)")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Average reward")
            ax.legend()
        fig.tight_layout()

        file_name = file_name or f"plot1_{bandit.name.lower().replace(' ', '_').replace('-', '_')}.png"
        fig.savefig(file_name, dpi=120)
        plt.close(fig)
        logger.info(f"plot1 for {bandit.name} saved to {file_name}")

    def plot2(self, eg, ts, file_name="plot2_comparison.png"):
        """
        Compare Epsilon-Greedy and Thompson Sampling cumulative rewards and regrets.

        :param eg: a finished EpsilonGreedy object
        :param ts: a finished ThompsonSampling object
        :param file_name: where to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for bandit in (eg, ts):
            axes[0].plot(np.cumsum(bandit.rewards), label=bandit.name)
            axes[1].plot(np.cumsum(bandit.regrets), label=bandit.name)

        axes[0].set_title("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        for ax in axes:
            ax.set_xlabel("Trial")
            ax.legend()
        axes[0].set_ylabel("Reward")
        axes[1].set_ylabel("Regret")
        fig.tight_layout()
        fig.savefig(file_name, dpi=120)
        plt.close(fig)
        logger.info(f"plot2 saved to {file_name}")

#--------------------------------------#


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm over several Gaussian arms.

    With probability epsilon_t = epsilon / t it explores (random arm),
    otherwise it exploits (arm with the highest estimated mean).
    """

    def __init__(self, p, epsilon=EPSILON, precision=PRECISION, seed=SEED):
        """
        :param p: list of true mean rewards, one per arm
        :param epsilon: starting epsilon, decayed by 1/t
        :param precision: precision of the reward noise (used only to simulate rewards)
        :param seed: random seed for reproducibility
        """
        self.name = "Epsilon-Greedy"
        self.p = list(p)
        self.k = len(self.p)
        self.epsilon = epsilon
        self.precision = precision
        self.rng = np.random.default_rng(seed)

        self.p_estimate = np.zeros(self.k)   # estimated mean of each arm
        self.N = np.zeros(self.k, dtype=int)  # times each arm was pulled
        self.chosen, self.rewards, self.regrets = [], [], []

    def __repr__(self):
        return f"EpsilonGreedy(p={self.p}, epsilon={self.epsilon})"

    def pull(self, arm):
        """
        Draw a reward from the chosen arm.

        :param arm: index of the arm
        :return: reward ~ N(p[arm], 1 / precision)
        """
        return self.rng.normal(self.p[arm], 1 / np.sqrt(self.precision))

    def update(self, arm, x):
        """
        Update the sample-mean estimate of the arm.

        :param arm: index of the arm
        :param x: observed reward
        """
        self.N[arm] += 1
        self.p_estimate[arm] += (x - self.p_estimate[arm]) / self.N[arm]

    def choose_arm(self, t):
        """
        Pick an arm with decaying epsilon.

        :param t: current trial (starting at 1)
        :return: index of the arm
        """
        eps_t = self.epsilon / t
        if self.rng.random() < eps_t:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.p_estimate))

    def experiment(self, n_trials=NUMBER_OF_TRIALS):
        """
        Run the experiment.

        :param n_trials: number of trials
        """
        best = max(self.p)
        for t in range(1, n_trials + 1):
            arm = self.choose_arm(t)
            x = self.pull(arm)
            self.update(arm, x)
            self.chosen.append(arm)
            self.rewards.append(x)
            self.regrets.append(best - self.p[arm])
        logger.debug(f"{self.name} finished {n_trials} trials")

    def report(self, csv_file=CSV_FILE):
        """
        Store the rewards in a CSV file and log the results.

        :param csv_file: CSV file the rows {Bandit, Reward, Algorithm} are appended to
        """
        df = pd.DataFrame({
            "Bandit": [self.p[a] for a in self.chosen],
            "Reward": self.rewards,
            "Algorithm": self.name,
        })
        df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)

        n = len(self.rewards)
        logger.info(f"[{self.name}] estimated means: {np.round(self.p_estimate, 3).tolist()}")
        logger.info(f"[{self.name}] pulls per arm: {self.N.tolist()}")
        logger.info(f"[{self.name}] cumulative reward: {np.sum(self.rewards):.2f}")
        logger.info(f"[{self.name}] cumulative regret: {np.sum(self.regrets):.2f}")
        logger.info(f"[{self.name}] average reward: {np.sum(self.rewards) / n:.4f}")
        logger.info(f"[{self.name}] average regret: {np.sum(self.regrets) / n:.4f}")

#--------------------------------------#


class ThompsonSampling(Bandit):
    """
    Thompson Sampling for Gaussian arms with known precision.

    Prior on each arm mean: N(m0, 1 / lambda0).
    After n rewards with sum s, the posterior is N(m, 1 / lambda) with
        lambda = lambda0 + n * tau
        m      = (lambda0 * m0 + tau * s) / lambda
    """

    def __init__(self, p, precision=PRECISION, prior_mean=0.0, prior_precision=1.0, seed=SEED):
        """
        :param p: list of true mean rewards, one per arm
        :param precision: known precision (tau) of the reward of each arm
        :param prior_mean: prior mean m0 of each arm
        :param prior_precision: prior precision lambda0 of each arm
        :param seed: random seed for reproducibility
        """
        self.name = "Thompson Sampling"
        self.p = list(p)
        self.k = len(self.p)
        self.tau = precision
        self.rng = np.random.default_rng(seed)

        self.m0, self.lambda0 = prior_mean, prior_precision
        self.m = np.full(self.k, prior_mean, dtype=float)        # posterior means
        self.lambda_ = np.full(self.k, prior_precision, dtype=float)  # posterior precisions
        self.sum_x = np.zeros(self.k)
        self.N = np.zeros(self.k, dtype=int)
        self.chosen, self.rewards, self.regrets = [], [], []

    def __repr__(self):
        return f"ThompsonSampling(p={self.p}, precision={self.tau})"

    def pull(self, arm):
        """
        Draw a reward from the chosen arm.

        :param arm: index of the arm
        :return: reward ~ N(p[arm], 1 / tau)
        """
        return self.rng.normal(self.p[arm], 1 / np.sqrt(self.tau))

    def sample(self):
        """
        Sample one value from each arm's posterior.

        :return: array of samples, one per arm
        """
        return self.rng.normal(self.m, 1 / np.sqrt(self.lambda_))

    def update(self, arm, x):
        """
        Update the posterior of the arm with a new reward.

        :param arm: index of the arm
        :param x: observed reward
        """
        self.N[arm] += 1
        self.sum_x[arm] += x
        self.lambda_[arm] = self.lambda0 + self.N[arm] * self.tau
        self.m[arm] = (self.lambda0 * self.m0 + self.tau * self.sum_x[arm]) / self.lambda_[arm]

    def experiment(self, n_trials=NUMBER_OF_TRIALS):
        """
        Run the experiment.

        :param n_trials: number of trials
        """
        best = max(self.p)
        for _ in range(n_trials):
            arm = int(np.argmax(self.sample()))
            x = self.pull(arm)
            self.update(arm, x)
            self.chosen.append(arm)
            self.rewards.append(x)
            self.regrets.append(best - self.p[arm])
        logger.debug(f"{self.name} finished {n_trials} trials")

    def report(self, csv_file=CSV_FILE):
        """
        Store the rewards in a CSV file and log the results.

        :param csv_file: CSV file the rows {Bandit, Reward, Algorithm} are appended to
        """
        df = pd.DataFrame({
            "Bandit": [self.p[a] for a in self.chosen],
            "Reward": self.rewards,
            "Algorithm": self.name,
        })
        df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)

        n = len(self.rewards)
        logger.info(f"[{self.name}] posterior means: {np.round(self.m, 3).tolist()}")
        logger.info(f"[{self.name}] pulls per arm: {self.N.tolist()}")
        logger.info(f"[{self.name}] cumulative reward: {np.sum(self.rewards):.2f}")
        logger.info(f"[{self.name}] cumulative regret: {np.sum(self.regrets):.2f}")
        logger.info(f"[{self.name}] average reward: {np.sum(self.rewards) / n:.4f}")
        logger.info(f"[{self.name}] average regret: {np.sum(self.regrets) / n:.4f}")


def comparison(eg, ts):
    """
    Compare the two algorithms visually and in numbers.

    Makes the plots and logs which algorithm got more reward and less regret.

    :param eg: a finished EpsilonGreedy object
    :param ts: a finished ThompsonSampling object
    """
    vis = Visualization()
    vis.plot1(eg)
    vis.plot1(ts)
    vis.plot2(eg, ts)

    for metric, values in [("reward", "rewards"), ("regret", "regrets")]:
        a, b = np.sum(getattr(eg, values)), np.sum(getattr(ts, values))
        logger.info(f"Cumulative {metric}: {eg.name} = {a:.2f} | {ts.name} = {b:.2f}")

    winner = eg if np.sum(eg.regrets) < np.sum(ts.regrets) else ts
    logger.success(f"{winner.name} has the lower cumulative regret")


if __name__ == '__main__':

    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        logger.warning(f"Old {CSV_FILE} removed")

    logger.info(f"Bandit rewards: {BANDIT_REWARD}, trials: {NUMBER_OF_TRIALS}")

    eg = EpsilonGreedy(BANDIT_REWARD)
    logger.debug(repr(eg))
    eg.experiment()
    eg.report()

    ts = ThompsonSampling(BANDIT_REWARD)
    logger.debug(repr(ts))
    ts.experiment()
    ts.report()

    comparison(eg, ts)
    logger.info(f"Rewards saved to {CSV_FILE}")
