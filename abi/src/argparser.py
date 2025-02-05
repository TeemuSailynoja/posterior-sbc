import argparse

import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to train.",
        choices=["m1a", "m2", "m3", "m4b", "m5", "m6"],
    )

    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default=f"UNNAMED_{np.random.randint(1e4):04d}",
    )

    parser.add_argument(
        "--nobs_fun",
        type=str,
        default="uniform",
        help="Function to generate the number of trials for each simulated data set.",
        choices=["uniform", "mixture"],
    )

    args = parser.parse_args(args=args)

    args.checkpoint_name = f"checkpoints/{args.checkpoint_prefix}_{args.model}"

    return args
