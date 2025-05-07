#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
from notears.linear import notears_linear
from notears.utils import count_accuracy

def main():
    p = argparse.ArgumentParser(
        description="Run NOTEARS on a single replicate dataset using the full data"
    )
    p.add_argument("--data-dir", required=True,
                   help="…/data_p<N>_e<E>_… directory containing data#.npy and DAG#.npy files")
    p.add_argument("--i-dataset", type=int, required=True,
                   help="Which replicate index to run (e.g. 1, 2, or 3)")
    p.add_argument("--lambda1",     type=float, default=0.1)
    p.add_argument("--loss-type",   choices=["l2","logistic","poisson"], default="l2")
    p.add_argument("--max-iter",    type=int,   default=100)
    p.add_argument("--h-tol",       type=float, default=1e-8)
    p.add_argument("--rho-max",     type=float, default=1e16)
    p.add_argument("--w-threshold", type=float, default=0.3)
    p.add_argument("--output-csv",  required=True,
                   help="Path to append results")
    args = p.parse_args()


    # prepare output CSV: write header if file doesn't exist
    header = ["dataset", "replicate", "runtime", "fdr", "tpr", "fpr", "shd", "nnz"]
    if not os.path.exists(args.output_csv):
        with open(args.output_csv, "w") as out:
            out.write(",".join(header) + "\n")


    # load only the specified replicate
    idx = args.i_dataset
    data_f = os.path.join(args.data_dir, f"data{idx}.npy")
    truth_f = os.path.join(args.data_dir, f"DAG{idx}.npy")

    X = np.load(data_f)                # shape (n, d)
    B_true = np.load(truth_f)          # {0,1} adjacency

    # run NOTEARS using the full dataset
    t0 = time.time()
    W_est = notears_linear(
        X,
        lambda1      = args.lambda1,
        loss_type    = args.loss_type,
        max_iter     = args.max_iter,
        h_tol        = args.h_tol,
        rho_max      = args.rho_max,
        w_threshold  = args.w_threshold
    )
    runtime = time.time() - t0

    # compute accuracy metrics
    acc = count_accuracy(B_true, (W_est != 0).astype(int))
    fdr = acc["fdr"]
    tpr = acc["tpr"]
    fpr = acc["fpr"]
    shd = acc["shd"]
    nnz = acc["nnz"]

    # append results to CSV
    with open(args.output_csv, "a") as out:
        out.write(",".join(map(str, [
            os.path.basename(args.data_dir),  # which simulation folder
            idx,                              # dataset index
            f"{runtime:.4f}",              # runtime (s)
            f"{fdr:.4f}",                  # false-discovery rate
            f"{tpr:.4f}",                  # true positive rate
            f"{fpr:.4f}",                  # false positive rate
            shd,                             # structural Hamming distance
            nnz                              # number of predicted edges
        ])) + "\n"
        )

if __name__ == "__main__":
    main()
