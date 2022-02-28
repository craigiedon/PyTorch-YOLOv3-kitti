import logging
import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import rc

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints


def simple_model(is_cont_africa, ruggedness, log_gdp=None):
    # MAP version - fixed params
    # a = pyro.param("a", lambda: torch.randn(()))
    # b_a = pyro.param("bA", lambda: torch.randn(()))
    # b_r = pyro.param("bR", lambda: torch.randn(()))
    # b_ar = pyro.param("bAR", lambda: torch.randn(()))
    # sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)

    # Bayesian version - distributions over params
    a = pyro.sample("a", dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.0))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.0))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    mean = a + (b_a * is_cont_africa) + (b_r * ruggedness) + (b_ar * is_cont_africa * ruggedness)

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)


def custom_guide(is_cont_africa, ruggedness, log_gdp=None):
    a_loc = pyro.param('a_loc', lambda: torch.tensor(0.))
    a_scale = pyro.param('a_scale', lambda: torch.tensor(1.0), constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', lambda: torch.tensor(1.0), constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', lambda: torch.randn(3))
    weights_scale = pyro.param('weights_scale', lambda: torch.ones(3), constraint=constraints.positive)

    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    return {"a": a, "b_a": b_a, "b_r": b_r, "b_ar": b_ar, "sigma": sigma}


def run():
    smoke_test = ('CI' in os.environ)
    assert pyro.__version__.startswith('1.8.0')

    pyro.enable_validation(True)
    pyro.set_rng_seed(1)
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Set Matplotlib settings
    plt.style.use('default')

    DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"

    data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df = data[["cont_africa", "rugged", "rgdppc_2000"]]

    df = df[np.isfinite(df.rgdppc_2000)]
    df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

    train = torch.tensor(df.values, dtype=torch.float)
    is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    # african_nations = df[df["cont_africa"] == 1]
    # non_african_nations = df[df["cont_africa"] == 0]
    # sns.scatterplot(x=non_african_nations["rugged"],
    #                 y=non_african_nations["rgdppc_2000"],
    #                 ax=ax[0])
    # ax[0].set(xlabel="Terrain Ruggedness Index",
    #           ylabel="log GDP (2000)",
    #           title="Non African Nations")
    # sns.scatterplot(x=african_nations["rugged"],
    #                 y=african_nations["rgdppc_2000"],
    #                 ax=ax[1])
    # ax[1].set(xlabel="Terrain Ruggedness Index",
    #           ylabel="log GDP (2000)",
    #           title="African Nations")
    # plt.show()

    pyro.render_model(simple_model, model_args=(is_cont_africa, ruggedness, log_gdp))
    pyro.render_model(custom_guide, model_args=(is_cont_africa, ruggedness, log_gdp)).view()


    pyro.clear_param_store()

    auto_guide = pyro.infer.autoguide.AutoNormal(simple_model)
    adam = pyro.optim.Adam({"lr": 0.02})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(simple_model, auto_guide, adam, elbo)

    losses = []
    for step in range(1000):
        loss = svi.step(is_cont_africa, ruggedness, log_gdp)
        losses.append(loss)
        if step % 100 == 0:
            logging.info("Elbo loss: {}".format(loss))

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.show()

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).data.cpu().numpy())

    with pyro.plate("samples", 800, dim=-1):
        samples = auto_guide(is_cont_africa, ruggedness)

    gamma_within_africa = samples["bR"] + samples["bAR"]
    gamma_outside_africa = samples["bR"]

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(gamma_within_africa.detach().cpu().numpy(), kde=True, stat="density", label="African nations")
    sns.histplot(gamma_outside_africa.detach().cpu().numpy(), kde=True, stat="density", label="Non-African nations",
                 color="orange")
    fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness");
    plt.xlabel("Slope of regression line")
    plt.legend()
    plt.show()

    predictive = pyro.infer.Predictive(simple_model, guide=auto_guide, num_samples=800)
    svi_samples = predictive(is_cont_africa, ruggedness, log_gdp=None)
    svi_gdp = svi_samples["obs"]

    mvn_guide = pyro.infer.autoguide.AutoMultivariateNormal(simple_model)
    pyro.render_model(mvn_guide, model_args=(is_cont_africa, ruggedness, log_gdp)).view()


if __name__ == "__main__":
    run()
