# %%
from pathlib import Path

import arviz as az
import jax
import jax.numpy as np
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi, print_summary
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpyro.infer import MCMC, NUTS, Predictive

# %%
az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

prng_key = random.PRNGKey(seed=42)

PATH_DATA = Path(__file__).parent.parent / "data"

# %%
df = pd.read_csv(PATH_DATA / "Howell1.csv", sep=";")

# %%
dfa = df.query("age > 18").copy()
dfa["weight"] -= dfa["weight"].mean()
dfa.sort_values(by="weight", inplace=True)
print_summary(dict(zip(dfa.columns, dfa.T.values)), 0.89, False)

# %%
priors = {
  "a": dist.Normal(175, 10),
  "b": dist.LogNormal(0, 1),
  "sigma": dist.Uniform(0, 50),
}

num_samples = int(1e3)

fig, ax = plt.subplots(3, 1)
i = 0
for n, d in priors.items():
  az.plot_kde(d.sample(prng_key, (num_samples,)), ax=ax[i])
  i += 1

# %%

def model(weight, height=None):
  a = numpyro.sample('a', priors['a'])
  b = numpyro.sample('b', priors['b'])
  sigma = numpyro.sample('sigma', priors['sigma'])
  mu = a + b * weight
  numpyro.sample("height_pred", dist.Normal(mu, sigma), obs=height)

# %%

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
kernel = NUTS(model)
num_samples = 2000
mcmc = MCMC(
  kernel, 
  num_warmup=1000, 
  num_samples=num_samples,     
  num_chains=4,
)
mcmc.run(
    rng_key_, 
    weight=dfa["weight"].values, 
    height=dfa["height"].values,
)
mcmc.print_summary()

# %%
num_samples_pred = 500
prior_pred_model = Predictive(model, num_samples=num_samples_pred)
prior_predictive = prior_pred_model(rng_key=prng_key, weight=dfa.weight.values)

posterior_samples = mcmc.get_samples()
posterior_predictive_model = Predictive(model, posterior_samples)
posterior_predictive = posterior_predictive_model(rng_key=prng_key, weight=dfa.weight.values)

samples = az.from_numpyro(
  posterior=mcmc, 
  prior=prior_predictive, 
  posterior_predictive=posterior_predictive,
)

print(az.summary(data=samples, hdi_prob=0.89).round(2))

az.plot_trace(data=samples)

# %%
dfa["height_pred"] = posterior_predictive["height_pred"].mean(axis=0)

# %%
sns.scatterplot(x=dfa["height"], y=dfa['height_pred'])

mse = ((dfa.height - dfa.height_pred)**2).mean(axis=0)
print(f'The RMSE in X_train is {np.sqrt(mse):.2f} cm.')

#%%
hdi_prob=0.95

height_pred_hdi = [
  az.hdi(ary=sample, hdi_prob=hdi_prob) for sample in
  posterior_predictive["height_pred"]._value.T
]

dfa[["height_hdi_low", "height_hdi_high"]] = height_pred_hdi
dfa.describe()
# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
sns.scatterplot(data=dfa, x='weight', y='height', alpha=0.5, ax=ax)
sns.lineplot(data=dfa, x='weight', y='height_pred', alpha=0.5, ax=ax)
ax.fill_between(
  dfa["weight"], 
  dfa["height_hdi_low"], 
  dfa["height_hdi_high"], 
  color='yellow', 
  alpha=0.3,
  interpolate=True,
)

# %%
def plot_regression(x, y_mean, y_hpdi):
    # Sort values for plotting by x axis
    idx = np.argsort(x)
    weight = x[idx]
    mean = y_mean[idx]
    hpdi = y_hpdi[:, idx]
    height = dfa["height"].values[idx]

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(weight, mean)
    ax.plot(weight, height, "o")
    ax.fill_between(weight, hpdi[0], hpdi[1], alpha=0.3, interpolate=True)
    return ax


mcmc_samples = mcmc.get_samples(2000)
post_mu = (
  np.expand_dims(mcmc_samples['a'].mean(axis=0), -1) 
  + np.expand_dims(mcmc_samples['b'].mean(axis=0), -1) * dfa.weight.values
)
mean_mu = np.mean(post_mu, axis=0)
hpdi_mu = hpdi(post_mu, 0.9)
ax = plot_regression(dfa.weight.values, mean_mu, hpdi_mu)

