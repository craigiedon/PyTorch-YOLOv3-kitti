import torch
from pyro.nn import PyroModule, PyroSample
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import pyro
import pyro.distributions as dist


class SalientObstacleDataset(Dataset):
    def __init__(self, s_inputs, s_labels):
        # Input Format:
        #   0: <Class Num>
        #   1: <Truncation>
        #   2: <Occlusion>
        #   3: <alpha>
        #   4-6: <dim_w> <dim_l> <dim_h>
        #   7-9: <loc_x> <loc_y> <loc_z>
        #   10: <rot_y>
        self.s_inp = s_inputs

        # Normalize 1, 3, 4,5,6,7,8,9,10
        norm_dims = [1, 3, 4, 5, 6, 7, 8, 9, 10]
        self.normed_ins, self.s_in_mus, self.s_in_stds = normalize_salient_data(s_inputs, norm_dims)

        # Pick Subset
        self.pose_subset = [7, 8, 9, 10]

        # One-hot encode 0, 2
        one_hot_cats = F.one_hot(self.s_inp[:, 0].to(dtype=torch.long), 7)
        one_hot_occlusion = F.one_hot(self.s_inp[:, 2].to(dtype=torch.long), 3)

        ## Final Indexing:
        # 0-6 Vehicle Cat One-hot
        # 7-9: Obscured One-hot
        # 10,11,12: x,y,z cam loc
        # 13: Rot y
        self.final_ins = torch.cat((one_hot_cats, one_hot_occlusion, self.normed_ins[:, self.pose_subset]), dim=1)
        assert self.final_ins.shape[0] == len(s_inputs)
        assert self.final_ins.shape[1] == 10 + len(self.pose_subset)

        # Label Format:
        #   0: <Detected>
        #   1-2: <bbox cx> <bbox cy>
        #   3-4: <bbox_w> <bbox_h>
        #   5-6: <err cx> <err cy>
        #   7-8: <err w> <err h>
        self.s_label = s_labels

    def __len__(self):
        return len(self.s_inp)

    def __getitem__(self, index):
        return self.final_ins[index], self.s_label[index]


def normalize_salient_data(s_inputs, norm_dims):
    in_mu = s_inputs.mean(0)
    in_std = s_inputs.std(0)

    normed_inputs = torch.detach(s_inputs)
    normed_inputs[:, norm_dims] = (normed_inputs[:, norm_dims] - in_mu[norm_dims]) / in_std[norm_dims]

    return normed_inputs, in_mu, in_std


class PEMClass_Deterministic(nn.Module):
    def __init__(self, in_d, out_d, h=20, use_cuda=False):
        super().__init__()
        self.ff_nn = nn.Sequential(
            nn.Linear(in_d, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, out_d),
        )

        if use_cuda:
            self.cuda()

    def forward(self, x):
        cat_logits = self.ff_nn(x)
        return cat_logits


class PEMClass(PyroModule):
    def __init__(self, in_d, out_d, h=20, use_cuda=False):
        super().__init__()

        self.t0 = torch.tensor(0.0)
        self.t1 = torch.tensor(1.0)

        if use_cuda:
            self.t0 = self.t0.cuda()
            self.t1 = self.t1.cuda()

        self.ff_1 = PyroModule[nn.Linear](in_d, h)
        self.ff_2 = PyroModule[nn.Linear](h, h)
        self.ff_3 = PyroModule[nn.Linear](h, out_d)

        self.ff_1.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([h, in_d]).to_event(2))
        self.ff_1.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([h]).to_event(1))

        self.ff_2.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([h, h]).to_event(2))
        self.ff_2.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([h]).to_event(1))

        self.ff_3.weight = PyroSample(dist.Normal(self.t0, self.t1).expand([out_d, h]).to_event(2))
        self.ff_3.bias = PyroSample(dist.Normal(self.t0, self.t1).expand([out_d]).to_event(1))

        if use_cuda:
            self.cuda()

    def forward(self, x, y=None):
        cat_logits = self.ff_1(x)
        cat_logits = F.relu(cat_logits)
        cat_logits = self.ff_2(cat_logits)
        cat_logits = F.relu(cat_logits)
        cat_logits = self.ff_3(cat_logits)

        with pyro.plate("data", x.shape[0], dim=-2):
            obs = pyro.sample("obs", dist.Bernoulli(logits=cat_logits), obs=y)
            # obs = pyro.sample("obs", dist.Categorical(logits=cat_logits), obs=y)

        return cat_logits


class PEMReg(PyroModule):
    def __init__(self, in_d, out_d, h=20):
        super().__init__()
        self.ff1_mu = PyroModule[nn.Linear](in_d, h)
        self.ff2_mu = PyroModule[nn.Linear](h, h)
        self.ff3_mu = PyroModule[nn.Linear](h, out_d)

        self.ff1_log_sig = PyroModule[nn.Linear](in_d, h)
        self.ff2_log_sig = PyroModule[nn.Linear](h, h)
        self.ff3_log_sig = PyroModule[nn.Linear](h, out_d)

    def forward(self, x, y=None):
        mu = self.ff1_mu(x)
        mu = F.relu(mu)
        mu = self.ff2_mu(mu)
        mu = F.relu(mu)
        mu = self.ff3_mu(mu)

        log_sig = self.ff1_log_sig(x)
        log_sig = F.relu(log_sig)
        log_sig = self.ff2_log_sig(log_sig)
        log_sig = F.relu(log_sig)
        log_sig = self.ff3_log_sig(log_sig)

        with pyro.plate("data", x.shape[0], dim=-2):
            obs = pyro.sample("obs", dist.Normal(mu, torch.exp(log_sig)), obs=y)

        return mu, log_sig


class PEMReg_Deterministic(nn.Module):
    def __init__(self, in_d, out_d, h=20, use_cuda=True):
        super().__init__()
        self.ff_nn = nn.Sequential(
            nn.Linear(in_d, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),

            nn.Linear(h, out_d),
        )

        if use_cuda:
            self.cuda()

    def forward(self, x):
        output = self.ff_nn(x)
        return output


def save_model_bnn(model_guide, param_store, f_name: str):
    param_store.save(f"{f_name}_guide_params.pt")
    torch.save(model_guide, f"{f_name}_guide.pt")


def load_model_bnn(guide_path: str, guide_params_path: str):
    guide = torch.load(guide_path)
    p_store = pyro.get_param_store()
    p_store.load(guide_params_path)
    return guide


def save_model_det(model, f_name: str):
    torch.save(model.state_dict(), f_name)


def load_model_det(model_skeleton, model_path: str):
    model_skeleton.load_state_dict(torch.load(model_path))
    model_skeleton.eval()
    return model_skeleton
