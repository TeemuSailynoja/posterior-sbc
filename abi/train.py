from functools import partial

import bayesflow as bf
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

if __name__ == "__main__":
    args = parse_args()

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
    print(num_params)
    amortizer = get_amortizer(cfg, num_params)

    trainer = bf.trainers.Trainer(
        amortizer=amortizer,
        generative_model=generative_model,
        configurator=configurator,
        default_lr=cfg.default_lr,
        checkpoint_path=args.checkpoint_name,
        max_to_keep=1,
    )

    h = trainer.train_online(
        epochs=cfg.epochs,
        iterations_per_epoch=cfg.iterations_per_epoch,
        batch_size=cfg.batch_size,
    )
