import importlib

import numpy as np


def get_prior(model_name: str) -> callable:
    """
    Returns the prior function for the specified model.
    """
    module = importlib.import_module(f".{model_name}", package="src.ddm")
    return module.prior


def get_batch_simulator(model_name: str) -> callable:
    """
    Returns the batch simulator function for the specified model.
    """
    module = importlib.import_module(f".{model_name}", package="src.ddm")
    return module.batch_simulator


def random_num_obs(num_obs_min: int = 200, num_obs_max: int = 700) -> int:
    """
    Returns a random number of observations between num_obs_min and num_obs_max.
    """
    n_obs = np.random.default_rng().integers(num_obs_min, num_obs_max + 1)
    return n_obs


def random_num_obs_mixture(num_obs_target: int = 288) -> int:
    """
    Returns a random number of observations with a mixture distribution with modes
    num_obs_target and 2*num_obs_target.
    """

    if np.random.default_rng().integers(0, 2) == 0:
        n_obs = np.random.default_rng().normal(loc=num_obs_target, scale=10)
    else:
        n_obs = np.random.default_rng().normal(loc=2 * num_obs_target, scale=10)
    return int(n_obs)


def configurator(forward_dict: dict) -> dict:
    out_dict = {}
    data = forward_dict["sim_data"].astype(np.float32)

    num_obs = forward_dict["sim_non_batchable_context"]
    vec_num_obs = np.ones((data.shape[0], 1)) * np.log(num_obs)  # transformed num_obs
    out_dict["direct_conditions"] = vec_num_obs.astype(np.float32)

    out_dict["parameters"] = forward_dict["prior_draws"].astype(np.float32)

    rt_signed = data[..., 0]
    cpp = data[..., 1]

    # recode RT as absolute_rt + response
    response = np.zeros_like(rt_signed, dtype=np.float32)
    response[rt_signed > 0] = 1.0

    rt_abs = np.abs(rt_signed)

    data_out = np.stack([rt_abs, response, cpp], axis=-1)

    out_dict["summary_conditions"] = data_out

    return out_dict
