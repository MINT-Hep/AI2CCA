import argparse
import os
import json
import numpy as np
import pandas as pd
import torch.multiprocessing
from transformers import AutoModel

from dataset import Generic_MIL_Dataset
from utils import get_simple_loader, seed_torch

from model_godin import GODINHead, GODINSequential
from engine import eval_odin
from ensemble_entropy import compute_predictive_entropy


torch.multiprocessing.set_start_method("spawn", force=True)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Pure testing for TITAN + G-ODIN (per-fold epsilon)")

    parser.add_argument("--data_root_dir", type=str, required=True, help="root directory for .h5 features")
    parser.add_argument("--csv_path", type=str, required=True, help="path to dataset CSV file")
    parser.add_argument("--models_dir", type=str, required=True, help="directory containing checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results", help="directory where results are stored")
    parser.add_argument("--exp_code", type=str, required=True, help="experiment code used in training")
    parser.add_argument("--eps_dir", type=str, required=True, help="directory containing epsilon_fold_{i}.json files")

    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--k", type=int, default=10, help="number of folds (default: 10)")
    parser.add_argument("--k_start", type=int, default=-1, help="starting fold index (default: -1 -> 0)")
    parser.add_argument("--k_end", type=int, default=-1, help="ending fold index (default: -1 -> k)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    # fallback ODIN params if JSON missing
    parser.add_argument("--odin_T", type=float, default=1.0, help="fallback temperature")
    parser.add_argument("--odin_eps", type=float, default=5e-5, help="fallback epsilon")

    args = parser.parse_args()
    seed_torch(device, args.seed)

    # === Dataset ===
    num_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_path,
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"Metastasis": 0, "ICCA": 1},
        patient_strat=True,
        ignore=[],
    )

    # results directory (match training structure)
    run_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    # folds
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    # model (TITAN + GODIN)
    backbone = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
    head = GODINHead(feat_dim=768, num_classes=num_classes).to(device)
    model = GODINSequential(backbone, head, return_scores=True).to(device)

    # metric containers
    acc, bacc, kappa, nw_kappa, weighted_f1, loss, auroc, fold_ids = [], [], [], [], [], [], [], []

    # === Loop over folds ===
    for i in folds:
        print(f"\n========== Fold {i} (TEST, best-eps) ==========\n")

        # build simple test loader for the entire dataset
        test_loader = get_simple_loader(dataset, batch_size=args.batch_size)

        # load model weights
        ckpt_path = os.path.join(args.models_dir, f"s_{i}_checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.to(device)
        model.eval()

        # load per-fold epsilon & temperature
        eps_path = os.path.join(args.eps_dir, f"epsilon_fold_{i}.json")
        if os.path.isfile(eps_path):
            with open(eps_path, "r") as f:
                eps_data = json.load(f)
            best_eps = float(eps_data.get("best_eps", args.odin_eps))
            odin_T = float(eps_data.get("T", args.odin_T))
            print(f"[FOLD {i}] Loaded best_eps={best_eps:g} from {eps_path}")
        else:
            best_eps = args.odin_eps
            odin_T = args.odin_T
            print(f"[FOLD {i}] WARN: {eps_path} missing, using fallback eps={best_eps:g}")

        # === Run ODIN evaluation ===
        results, outputs, _ = eval_odin(
            loader=test_loader,
            model=model,
            num_classes=num_classes,
            device=device,
            T=odin_T,
            eps=best_eps,
        )

        # record metrics
        fold_ids.append(i)
        acc.append(results.get("acc", -1))
        bacc.append(results.get("bacc", -1))
        kappa.append(results.get("kappa", -1))
        nw_kappa.append(results.get("nw_kappa", -1))
        weighted_f1.append(results.get("weighted_f1", -1))
        loss.append(results.get("loss", -1))
        auroc.append(results.get("auroc", -1))

        # save per-fold predictions
        if num_classes == 2:
            df_fold = pd.DataFrame(outputs)
            df_fold["p_0"] = 1 - df_fold["probs"].values
            df_fold["p_1"] = df_fold["probs"].values
            df_fold = df_fold.drop(columns="probs")
        else:
            for n in range(num_classes):
                outputs[f"p_{n}"] = outputs["probs"][:, n]
            outputs.pop("probs", None)
            df_fold = pd.DataFrame(outputs)

        out_csv = os.path.join(run_dir, f"fold_{i}_odin.csv")
        df_fold.to_csv(out_csv, index=False)
        print(f"Saved ODIN predictions for fold {i} to {out_csv}")

    # === summary across folds ===
    df_all = pd.DataFrame(
        {
            "fold": fold_ids,
            "acc": acc,
            "bacc": bacc,
            "kappa": kappa,
            "nw_kappa": nw_kappa,
            "weighted_f1": weighted_f1,
            "loss": loss,
            "auroc": auroc,
        }
    )

    save_name = "summary.csv" if len(folds) == args.k else f"summary_partial_{start}_{end}.csv"
    summary_path = os.path.join(run_dir, save_name)
    df_all.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to {summary_path}")

    # === compute predictive entropy across all folds ===
    csv_paths = [os.path.join(run_dir, f"fold_{i}_odin.csv") for i in folds]
    csv_paths = [p for p in csv_paths if os.path.isfile(p)]
    if len(csv_paths) == 0:
        print("No per-fold CSVs found; skipping entropy computation.")
    else:
        entropy_out_path = os.path.join(run_dir, "ensemble_predictive_entropy.xlsx")
        df_entropy = compute_predictive_entropy(csv_paths, entropy_out_path)
        print(f"\nSaved predictive entropy summary to {entropy_out_path}")
        print(df_entropy.head())

