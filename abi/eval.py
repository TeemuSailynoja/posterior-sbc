import os
from functools import partial

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.argparser import parse_args
from src.config import cfg
from src.ddm import (
    configurator,
    get_batch_simulator,
    get_prior,
    random_num_obs,
    random_num_obs_mixture,
)
from src.models import get_amortizer
from src.posterior_sbc import posterior_sbc

# set working directory to root of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def preprocess_data(df):
    df["response_corr"] = df["response_corr"].replace(0, -1)
    df = np.array([df["response_time"] * df["response_corr"], df["n200lat"]]).T
    df = df[df[:, 1] > -10]

    df = df[: cfg.num_test_observations]

    return df


subject_indices = [1, 3, 6]

real_data = {
    subject_idx: preprocess_data(
        pd.read_csv(
            f"data/sub-{subject_idx:03}_task-pdm_acq-outsideMRT_runs_beh_n200lat.csv"
        )
    )
    for subject_idx in subject_indices
}

print([real_observations.shape for real_observations in real_data.values()])


if __name__ == "__main__":
    args = parse_args()

    args.plot_path = f"plots/{args.checkpoint_prefix}_{args.model}"
    os.makedirs(args.plot_path, exist_ok=True)

    param_names = cfg.param_names[args.model]

    num_params = len(param_names)

    if args.nobs_fun == "uniform":
        context_gen = bf.simulation.ContextGenerator(
            non_batchable_context_fun=partial(
                random_num_obs, num_obs_min=cfg.num_obs_min, num_obs_max=cfg.num_obs_max
            )
        )
    elif args.nobs_fun == "mixture":
        context_gen = bf.simulation.ContextGenerator(
            non_batchable_context_fun=partial(
                random_num_obs_mixture, num_obs_target=cfg.num_test_observations
            )
        )
    else:
        raise ValueError("Invalid nobs_fun")

    prior = bf.simulation.Prior(batch_prior_fun=get_prior(args.model))
    simulator = bf.simulation.Simulator(
        batch_simulator_fun=get_batch_simulator(args.model),
        context_generator=context_gen,
    )
    generative_model = bf.simulation.GenerativeModel(prior=prior, simulator=simulator)

    amortizer = get_amortizer(cfg, num_params)

    trainer = bf.trainers.Trainer(
        amortizer=amortizer,
        generative_model=generative_model,
        configurator=configurator,
        checkpoint_path=args.checkpoint_name,
        max_to_keep=1,
    )

    # Loss history
    f = bf.diagnostics.plot_losses(trainer.loss_history.get_plottable())
    f.savefig(f"{args.plot_path}_loss_history.png")
    plt.close()

    # Closed-world evaluation on data from the joint model used for training
    # Evaluate on n_obs equal to the number of observations in the real data, and
    # also on 2N observations because we to evaluate on 2N for PosteriorSBC
    for label, num_obs in {
        "N": cfg.num_test_observations,
        "2N": 2 * cfg.num_test_observations,
    }.items():
        theta_true = get_prior(args.model)(cfg.num_test_datasets)
        y_true = get_batch_simulator(
            args.model
        )(
            theta_true, num_obs
        )  # For each drawn paramter vector, sample num_obs observations from the simulator
        test_data = trainer.configurator(
            {
                "prior_draws": theta_true,
                "sim_data": y_true,
                "sim_non_batchable_context": num_obs,
            }
        )

        posterior_samples = trainer.amortizer.sample(
            test_data, n_samples=cfg.num_posterior_samples
        )

        # PriorSBC
        f = bf.diagnostics.plot_sbc_ecdf(
            posterior_samples, theta_true, difference=True, stacked=True
        )
        f.savefig(f"{args.plot_path}_priorsbc_{label}.png")
        plt.close()

        # Recovery of the true parameters
        f = bf.diagnostics.plot_recovery(
            posterior_samples,
            theta_true,
            param_names=param_names,
            add_r2=False,
        )
        f.savefig(f"{args.plot_path}_recovery_{label}.png")
        plt.close()

    for subject_idx, y_obs in real_data.items():
        # Open-world evaluation on real data
        # PosteriorSBC
        posterior_samples_y, conditional_posterior_samples = posterior_sbc(
            y_obs=y_obs,
            trainer=trainer,
            ppred_simulator=get_batch_simulator(args.model),
            num_ppred_samples=200,
            num_posterior_samples=500,
        )

        f = bf.diagnostics.plot_sbc_ecdf(
            conditional_posterior_samples,
            posterior_samples_y,
            difference=True,
            stacked=True,
        )
        f.savefig(f"{args.plot_path}_posteriorsbc_{subject_idx}.png")

        print(f"Done with subject {subject_idx}")

    print("Done with all subjects")
