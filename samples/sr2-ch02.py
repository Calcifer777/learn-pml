# %%
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.stats import binom

# %%
def uniform_prior(x):
    return np.repeat(x.max() - x.min(), x.shape[0])


def truncated_prior(x, threshold=0.5):
    return (x >= threshold).astype(int) / (x.max() - threshold)


def double_exp_prior(x):
    return np.exp(-5 * abs(x - 0.5))


# %%
def posterior(prior, likelihood, data):
  posterior_values = prior(data) * likelihood(data)
  return posterior_values / posterior_values.sum()


# %%
x = np.linspace(0, 1, 200)
n = 10
successes = 6

# %%
posterior_values = posterior(
  data = x,
  prior=uniform_prior,
  likelihood=(lambda x: binom.pmf(successes, n, x))
)
plt.plot(x, posterior_values)

# %%
posterior_values = posterior(
  data = x,
  prior=truncated_prior,
  likelihood=(lambda x: binom.pmf(successes, n, x))
)
plt.plot(x, posterior_values)

# %%
posterior_values = posterior(
  data = x,
  prior=double_exp_prior,
  likelihood=(lambda x: binom.pmf(successes, n, x))
)
plt.plot(x, posterior_values)

# %%
# Quadratic approximation
data = np.repeat((0, 1), (n-successes, successes))

with pm.Model() as normal_approximation:
    prior = pm.Uniform("p", 0, 1)  # uniform priors
    likelihood = pm.Binomial("w", n=len(data), p=prior, observed=data.sum())  # binomial likelihood
    mean_q = pm.find_MAP()

    # p_value = normal_approximation.rvs_to_values[prior]
    # p_value.tag.transform = None
    # p_value.name = prior.name

    # std_q = ((1 / pm.find_hessian(mean_q, vars=[prior])) ** 0.5)[0]

# display summary of quadratic approximation
# print("Mean, Standard deviation\np {:.2}, {:.2}".format(mean_q["p"], std_q[0]))

# %%
