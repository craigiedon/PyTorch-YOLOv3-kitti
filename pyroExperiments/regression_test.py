import numpy as np
import pyro
import torch.nn
from matplotlib import pyplot as plt
from torch import nn
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
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


def run():
    x_data = torch.tensor(np.linspace(0, 2 * np.pi, 100), dtype=torch.float).view(-1, 1)  # .view((-1, 1))
    fx_data = 0.5 * torch.sin(x_data) - 0.5 * torch.cos(3 * x_data)  # (x_data ** 2) * -0.01
    y_data = fx_data + 0.2 * torch.randn((len(x_data), 1))

    # linear_reg_model = PyroModule[nn.Linear](1, 1)
    # loss_fn = torch.nn.MSELoss(reduction='sum')
    # optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
    num_iterations = 1500

    model = BayesianRegression(1, 1)
    guide = AutoDiagonalNormal(model)
    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    for j in range(num_iterations):
        loss = svi.step(x_data, y_data.squeeze(-1))
        if j % 100 == 0:
            print("[iteration %d] loss: %.4f" % (j + 1, loss / len(x_data)))

    guide.requires_grad_(False)

    # pyro.render_model(model, model_args=(x_data, y_data.squeeze(-1))).view()
    # pyro.render_model(guide).view()

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    predictive = Predictive(model, guide=guide, num_samples=800, return_sites=("linear.weight", "obs", "_RETURN"))
    samples = predictive(x_data)
    pred_summary = summary(samples)

    plt.plot(x_data.detach().numpy(), fx_data.detach().numpy(), color='orange')
    plt.scatter(x_data.detach().numpy(), y_data.detach().numpy(), marker='x', color='green', alpha=0.5)
    # plt.plot(x_data.detach().numpy(), pred_summary["obs"]["mean"], color='b')
    plt.plot(x_data.detach().numpy(), pred_summary["_RETURN"]["mean"], color='b')
    # plt.fill_between(x_data.detach().numpy()[:, 0], pred_summary["_RETURN"]["5%"], pred_summary["_RETURN"]["95%"], alpha=0.5)
    # plt.fill_between(x_data.detach().numpy()[:, 0], pred_summary["_RETURN"]["5%"], pred_summary["_RETURN"]["95%"], alpha=0.5)
    plt.fill_between(x_data.detach().numpy()[:, 0],
                     pred_summary["obs"]["mean"] - pred_summary["obs"]["std"],
                     pred_summary["obs"]["mean"] + pred_summary["obs"]["std"],
                     alpha=0.4)
    plt.show()


if __name__ == "__main__":
    run()
