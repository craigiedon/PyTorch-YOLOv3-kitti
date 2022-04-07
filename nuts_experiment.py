import torch
import numpy as np
from pyro.infer import NUTS, MCMC, Predictive
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score, \
    balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader

from full_tp_pem import k_fold_validaton_bnn
from pems import SalientObstacleDataset, PEMClass


def run():
    s_inputs = torch.tensor(np.loadtxt("salient_dataset/salient_inputs_no_miscUnknown.txt"), dtype=torch.float)
    s_labels = torch.tensor(np.loadtxt("salient_dataset/salient_labels_no_miscUnknown.txt"), dtype=torch.float)
    full_data_set = SalientObstacleDataset(s_inputs, s_labels)

    # k_fold_validaton_bnn(full_data_set, 5, train_model, "bnn")
    train_model_nuts(full_data_set, "saved_models/full_set/")
    eval_nuts(torch.load("saved_models/nuts/nuts_samples.pt"), full_data_set)


def train_model_nuts(train_set: SalientObstacleDataset, results_folder: str):
    print(results_folder)
    print(f"Training Set Size: {len(train_set)}")

    model = PEMClass(14, 1, 20, use_cuda=True)

    nuts_kernel = NUTS(model, jit_compile=True, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    skip = 3
    train_ins = train_set.final_ins[::skip].cuda()
    train_label = train_set.s_label[::skip, 0].cuda()

    mcmc.run(train_ins, train_label)
    samples = mcmc.get_samples()
    torch.save(samples, "saved_models/nuts/nuts_samples.pt")
    # summary = mcmc.summary()

    # save_model_bnn(guide, pyro.get_param_store(), join(results_folder, "pem_class_train_full"))
    return samples


def eval_nuts(samples, test_set):
    combined_preds = torch.tensor([]).cuda()
    combined_trus = torch.tensor([]).cuda()

    model = PEMClass(14, 1, 5, use_cuda=True)

    test_loader = DataLoader(test_set, 8000)
    test_predictive = Predictive(model, posterior_samples=samples, return_sites={"obs", "_RETURN"})

    for test_x, test_y in test_loader:
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        pred_test_y = torch.sigmoid(test_predictive(test_x)["_RETURN"]).mean(0)

        combined_preds = torch.cat((combined_preds, pred_test_y))
        combined_trus = torch.cat((combined_trus, test_y[:, [0]]))

    combined_preds = combined_preds.detach().cpu().numpy()
    combined_trus = combined_trus.detach().cpu().numpy()

    fpr, tpr, ft_threshs = roc_curve(combined_trus, combined_preds)
    prec, rec, pr_threshs = precision_recall_curve(combined_trus, combined_preds)

    bce = F.binary_cross_entropy(torch.tensor(combined_preds), torch.tensor(combined_trus)).item()

    auc = roc_auc_score(combined_trus, combined_preds)
    avg_p = average_precision_score(combined_trus, combined_preds)
    accuracy = accuracy_score(combined_trus, np.round(combined_preds))
    b_accuracy = balanced_accuracy_score(combined_trus, np.round(combined_preds))

    print(f"AUC {auc}, Avg_p: {avg_p}, Accuracy: {accuracy}, b_accuracy: {b_accuracy}")

    return {"roc_curve": (fpr, tpr, ft_threshs),
            "pr_curve": (list(reversed(rec)), list(reversed(prec)), pr_threshs),
            "auc": auc,
            "avg_p": avg_p,
            "accuracy": accuracy,
            "b_accuracy": b_accuracy,
            "bce": bce
            }


if __name__ == "__main__":
    run()
