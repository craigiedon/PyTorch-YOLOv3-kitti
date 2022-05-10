from collections import defaultdict
from os.path import join

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from full_reg_pem import eval_model, eval_model_al, guess_mu_baseline, linear_baseline
from pems import PEMReg, load_model_bnn, PEMReg_Aleatoric, load_model_det, SalientObstacleDataset


def run():
    s_inputs = torch.tensor(np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt"), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt"), dtype=torch.float)
    full_data_set = SalientObstacleDataset(s_inputs, s_labels)
    eval_kfold_models(full_data_set, 5)


def eval_kfold_models(dataset, k):
    kf = KFold(n_splits=k)

    baseline_metrics = defaultdict(list)

    for ki, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Fold: {ki}")
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        bnn_results_folder = f"saved_models/bnn_reg_k{ki}"
        kth_model = PEMReg(14, 2, use_cuda=True)
        kth_guide = load_model_bnn(join(bnn_results_folder, "pem_reg_train_full_guide.pt"),
                                   join(bnn_results_folder, "pem_reg_train_full_guide_params.pt"))

        det_model = load_model_det(PEMReg_Aleatoric(14, 2, use_cuda=True),
                                   f"saved_models/al_reg_k{ki}/pem_reg_al_full")

        baseline_metrics["LR"].append(linear_baseline(train_set, test_set))
        baseline_metrics["guess_mu"].append(guess_mu_baseline(train_set, test_set))
        baseline_metrics["Aleatoric-NN"].append(eval_model_al(det_model, test_set))
        baseline_metrics["B-NN"].append(eval_model(kth_model, kth_guide, test_set))

        print(baseline_metrics)

    for k, v in baseline_metrics.items():
        print(k, np.average(v))


if __name__ == "__main__":
    run()
