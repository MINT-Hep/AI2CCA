import argparse
import os
import numpy as np
import pandas as pd
import builtins
import torch.multiprocessing
from transformers import AutoModel
from dataset import Generic_MIL_Dataset

from utils import get_split_loader, seed_torch

from model_godin import GODINHead, GODINSequential
from engine import train, eval, search_epsilon, eval_odin


torch.multiprocessing.set_start_method("spawn", force=True)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Finetune TITAN + per-fold G-ODIN epsilon search")

    # Training / optimization
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--accum_steps", type=int, default=16, help="gradient accumulation steps")

    # Data
    parser.add_argument("--data_root_dir", type=str, required=True, help="root directory for patch features (.h5)")
    parser.add_argument("--csv_path", type=str, required=True, help="path to the main CSV file")
    parser.add_argument("--split_dir", type=str, required=True, help="directory that contains split CSVs (splits_*.csv)")

    # CV
    parser.add_argument("--k", type=int, default=10, help="number of folds (default: 10)")
    parser.add_argument("--k_start", type=int, default=-1, help="start fold index (default: -1 -> 0)")
    parser.add_argument("--k_end", type=int, default=-1, help="end fold index (default: -1 -> k)")

    # IO
    parser.add_argument("--results_dir", default="./results", help="results directory (default: ./results)")
    parser.add_argument("--exp_code", type=str, required=True, help="experiment code for saving results")
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiments")

    # Loader options
    parser.add_argument(
        "--weighted_sample",
        action="store_true",
        default=True,
        help="enable weighted sampling in the training loader",
    )

    # ODIN search
    parser.add_argument("--odin_T", type=float, default=1.0, help="temperature for G-ODIN")
    parser.add_argument(
        "--eps_grid",
        type=str,
        default="0,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3",
        help="comma-separated epsilon grid; supports scientific notation like 5e-5",
    )
    parser.add_argument(
        "--allow_acc_drop",
        type=float,
        default=0.005,
        help="max absolute accuracy drop allowed on VAL vs eps=0 (e.g., 0.005 = 0.5%)",
    )

    args = parser.parse_args()

    # Parse epsilon grid string -> floats
    def _parse_eps_grid(s: str):
        toks = [t.strip() for t in s.split(",") if t.strip()]
        vals = []
        for t in toks:
            vals.append(float(builtins.eval(t)))  # allow sci-notation in string
        return vals

    eps_list = _parse_eps_grid(args.eps_grid)

    # Set seeds
    seed_torch(device, args.seed)

    # ---------------------------------------------------------------------
    # Load dataset (Metastasis vs ICCA)
    # ---------------------------------------------------------------------
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

    # Prepare results directory
    args.results_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}")
    os.makedirs(args.results_dir, exist_ok=True)

    # Determine folds to run
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    acc, auc, fpr, loss, fold_ids = [], [], [], [], []

    # ---------------------------------------------------------------------
    # Cross-validation loop
    # ---------------------------------------------------------------------
    for i in folds:
        print(f"\n========== Fold {i} ==========\n")

        # Load TITAN backbone
        backbone = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)

        # Add GODIN head and wrap
        head = GODINHead(feat_dim=768, num_classes=num_classes).to(device)
        model = GODINSequential(backbone, head, return_scores=True).to(device)

        # Get train / val / test splits for this fold
        split_csv = os.path.join(args.split_dir, f"splits_{i}.csv")
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False,
            csv_path=split_csv
        )

        # Data loaders
        train_loader = get_split_loader(
            train_dataset,
            training=True,
            weighted=args.weighted_sample,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        val_loader = get_split_loader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        test_loader = get_split_loader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # ------------------ Train ------------------
        model = train(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            accum_steps=args.accum_steps,
        )

        # Save checkpoint
        ckpt_path = os.path.join(args.results_dir, f"s_{i}_checkpoint.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # ------------------ ODIN eps search on VAL ------------------
        eps_json = os.path.join(args.results_dir, f"epsilon_fold_{i}.json")
        best_eps, base_acc, base_mean_pmax = search_epsilon(
            val_loader=val_loader,
            model=model,
            num_classes=num_classes,
            device=device,
            T=args.odin_T,
            eps_list=eps_list,
            allow_acc_drop=args.allow_acc_drop,
            json_path=eps_json,
        )
        print(f"[Fold {i}] Best epsilon on VAL: {best_eps:g}  (baseline acc={base_acc:.4f}, mean_pmax={base_mean_pmax:.6f})")
        print(f"[Fold {i}] Epsilon selection saved to {eps_json}")

        # ------------------ Test (best eps) ------------------
        # ------------------ Test (ODIN with T, best eps) ------------------
        odin_metrics, odin_outputs, _ = eval_odin(
            loader=test_loader,
            model=model,
            num_classes=num_classes,
            device=device,
            T=args.odin_T,
            eps=best_eps,
        )

        acc.append(odin_metrics["acc"])
        auc.append(odin_metrics["auc"])
        fpr.append(odin_metrics["fpr"])
        loss.append(odin_metrics["loss"])
        fold_ids.append(i)

        # Save ODIN best-eps predictions
        if num_classes == 2:
            df_fold_odin = pd.DataFrame(odin_outputs)
            df_fold_odin["p_0"] = 1.0 - df_fold_odin["probs"].values
            df_fold_odin["p_1"] = df_fold_odin["probs"].values
            df_fold_odin = df_fold_odin.drop(columns="probs")
        else:
            for n in range(num_classes):
                col = f"p_{n}"
                odin_outputs[col] = odin_outputs["probs"][:, n]
            odin_outputs.pop("probs", None)
            df_fold_odin = pd.DataFrame(odin_outputs)

        fold_csv_odin = os.path.join(args.results_dir, f"fold_{i}_odin.csv")
        df_fold_odin.to_csv(fold_csv_odin, index=False)
        print(f"Saved fold predictions (eps={best_eps:g}) to {fold_csv_odin}")

    df_all = pd.DataFrame(
        {
            "fold": fold_ids,
            "acc": acc,
            "auc": auc,
            "fpr": fpr,
            "loss": loss,
        }
    )

    save_name = "summary.csv" if len(folds) == args.k else f"summary_partial_{start}_{end}.csv"
    summary_path = os.path.join(args.results_dir, save_name)
    df_all.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to {summary_path}")
