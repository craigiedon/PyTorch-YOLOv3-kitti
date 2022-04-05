import numpy as np
import pyro
from pyro import poutine
from pyro.infer import MCMC, NUTS, TraceGraph_ELBO
import torch.nn
from matplotlib import pyplot as plt
from torch import nn
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive


class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


class Model(PyroModule):
    def __init__(self, h1=20, h2=20, use_cuda=False):
        super().__init__()

        if use_cuda:
            self.t0 = torch.tensor(0.0).cuda()
            self.t1 = torch.tensor(1.0).cuda()
        else:
            self.t0 = 0.0
            self.t1 = 1.0

        self.fc1 = PyroModule[nn.Linear](1, h1)
        self.fc1.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([h1, 1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([h1]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([h1, h2]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([h2]).to_event(1))

        self.fc3 = PyroModule[nn.Linear](h2, 1)
        self.fc3.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([1, h2]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([1]).to_event(1))

        self.fc1_log_sigma = PyroModule[nn.Linear](1, h1)
        self.fc1_log_sigma.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([h1, 1]).to_event(2))
        self.fc1_log_sigma.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([h1]).to_event(1))

        self.fc2_log_sigma = PyroModule[nn.Linear](h1, h2)
        self.fc2_log_sigma.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([h1, h2]).to_event(2))
        self.fc2_log_sigma.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([h2]).to_event(1))

        self.fc3_log_sigma = PyroModule[nn.Linear](h2, 1)
        self.fc3_log_sigma.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([1, h2]).to_event(2))
        self.fc3_log_sigma.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([1]).to_event(1))

        self.relu = nn.ReLU()

        if use_cuda:
            self.cuda()

    def forward(self, x, y=None):
        x = x.view(-1, 1)

        mu = self.relu(self.fc1(x))
        mu = self.relu(self.fc2(mu))
        mu = self.fc3(mu).squeeze()

        log_sigma = self.relu(self.fc1_log_sigma(x))
        log_sigma = self.relu(self.fc2_log_sigma(log_sigma))
        log_sigma = self.fc3_log_sigma(log_sigma)
        sigma = torch.exp(log_sigma).squeeze()

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        return mu


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


def trans_func(xs):
    return 0.5 * torch.sin(xs) - 0.5 * torch.cos(3 * xs)


def run():
    x_data = torch.tensor(np.linspace(0, 2 * np.pi, 1000), dtype=torch.float)
    x_mean = x_data.mean()
    x_std = x_data.std()
    x_data = (x_data - x_mean) / x_std
    print(x_mean, x_std)
    fx_data = trans_func(x_data)
    y_data = fx_data + 0.2 * torch.cat((torch.randn(len(x_data) // 2), torch.zeros(len(x_data) // 2)))

    # linear_reg_model = PyroModule[nn.Linear](1, 1)
    # loss_fn = torch.nn.MSELoss(reduction='sum')
    # optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
    num_iterations = 20000

    model = Model(20, 20, False)
    guide = AutoDiagonalNormal(model)
    p_store = pyro.get_param_store()

    # nuts_kernel = NUTS(model, jit_compile=False)
    # mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=200, num_chains=1)
    # mcmc.run(x_data, y_data)
    #
    # hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    # print(hmc_samples)

    adam = pyro.optim.Adam({"lr": 1e-3})
    svi = SVI(model, guide, adam, loss=Trace_ELBO(vectorize_particles=True))

    pyro.clear_param_store()
    pyro.render_model(model, model_args=(x_data, y_data.squeeze(-1))).view()
    pyro.render_model(guide).view()

    for j in range(num_iterations):
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            print("[iteration %d] loss: %.4f" % (j + 1, loss / len(x_data)))

    guide.requires_grad_(False)

    pyro.get_param_store().save("../saved_models/guide_params.save")
    torch.save(guide, "../saved_models/guide.pt")

    print("Parameter names")
    for name, value in pyro.get_param_store().items():
        # print(name, pyro.param(name))
        print(name)

    predictive = Predictive(model, guide=guide, num_samples=2000)
    x_pred_points = (torch.linspace(-2 * np.pi, 4 * np.pi, 1000) - x_mean) / x_std
    preds = predictive(x_pred_points)
    # pred_summary = summary(samples)

    y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
    y_std = preds['obs'].T.detach().numpy().std(axis=1)

    plt.scatter(x_data.detach().numpy(), y_data.detach().numpy(), marker='x', color='green', alpha=0.2)
    plt.plot(x_pred_points.detach().numpy(), trans_func(x_pred_points).detach().numpy(), color='orange')
    plt.plot(x_pred_points.detach().numpy(), y_pred, color='b')
    plt.fill_between(x_pred_points, y_pred - y_std, y_pred + y_std, alpha=0.5, color='b')
    plt.ylim(-5, 5)

    # plt.plot(x_data.detach().numpy(), pred_summary["obs"]["mean"], color='b')
    # plt.plot(x_data.detach().numpy(), pred_summary["_RETURN"]["mean"], color='b')
    # plt.fill_between(x_data.detach().numpy()[:, 0], pred_summary["_RETURN"]["5%"], pred_summary["_RETURN"]["95%"], alpha=0.5)
    # plt.fill_between(x_data.detach().numpy()[:, 0], pred_summary["_RETURN"]["5%"], pred_summary["_RETURN"]["95%"], alpha=0.5)
    # plt.fill_between(x_data.detach().numpy()[:, 0],
    #                  pred_summary["obs"]["mean"] - pred_summary["obs"]["std"],
    #                  pred_summary["obs"]["mean"] + pred_summary["obs"]["std"],
    #                  alpha=0.4)
    plt.show()


if __name__ == "__main__":
    run()
