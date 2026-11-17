# Homework 2 – A/B Testing with Multi-Armed Bandits

Four ads (bandits) with mean rewards `[1, 2, 3, 4]`, tested for 20,000 trials with
**Epsilon-Greedy** and **Thompson Sampling**.

## How to run
```bash
pip install -r requirements.txt
python Bandit.py
```

## Design choices
- Rewards: each ad gives a Gaussian reward, mean = its Bandit_Reward value, precision = 1.
- Epsilon-Greedy: epsilon starts at 1 and decays as `1/t`.
- Thompson Sampling: known precision `tau = 1`, prior `N(0, 1)` on each mean (Normal–Normal conjugate update).
- Regret per trial = best mean (4) − true mean of the chosen ad.
- `print()` is replaced with `loguru` log levels.

## Outputs
| File | What it shows |
|---|---|
| `rewards.csv` | `{Bandit, Reward, Algorithm}` for every trial |
| `plot1_epsilon_greedy.png` | E-Greedy learning process (linear + log) |
| `plot1_thompson_sampling.png` | Thompson learning process (linear + log) |
| `plot2_comparison.png` | Cumulative reward and regret of both algorithms |

## Results (seed = 42)
| Algorithm | Cumulative reward | Cumulative regret |
|---|---|---|
| Epsilon-Greedy | 73,907 | 6,181 |
| Thompson Sampling | 79,967 | 7 |

Thompson Sampling finds the best ad almost right away. Epsilon-Greedy got stuck on ad 3
for about 6,000 trials, because `1/t` makes epsilon very small very fast, so it almost stops exploring.

## Bonus – a better implementation plan
1. **Better epsilon schedule** – use `epsilon / t` with a floor (e.g. `max(0.01, ...)`) or a slower decay like `c / sqrt(t)` so E-Greedy does not lock onto a wrong ad.
2. **Optimistic start** – set initial estimates high (e.g. 10) so every ad gets tried a few times before exploiting.
3. **Try every ad once first** – a short warm-up phase before the algorithm starts.
4. **UCB1** – add Upper Confidence Bound as a third algorithm; it explores in a smarter way than random.
5. **Unknown precision** – use a Normal-Gamma prior in Thompson Sampling so the noise level is learned, not assumed.
6. **Many runs** – repeat each experiment with different seeds and plot the average (with a band), so results do not depend on one lucky or unlucky run.
7. **Cleaner code** – move shared logic (`pull`, `report`) into a base class, keep settings in a config file, and add unit tests.
