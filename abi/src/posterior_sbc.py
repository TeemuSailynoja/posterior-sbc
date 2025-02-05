import numpy as np


def posterior_sbc(
    y_obs, trainer, ppred_simulator, num_ppred_samples=200, num_posterior_samples=500
):
    """
    y_obs:      np.array
                observed data, shape (num_obs, data_dim)

    trainer:    bf.trainers.Trainer

    ppred_simulator: callable

    num_ppred_samples: int, default: 200
                number of samples from the posterior predictive distribution

    num_posterior_sampels: int, default: 500
                number of ("conditional") posterior samples to draw per ppred sample
    """

    num_obs, data_dim = y_obs.shape

    num_params = trainer.amortizer.inference_net.latent_dim

    y_obs_configured = trainer.configurator(
        {
            "sim_data": np.array(y_obs, dtype=np.float32)[np.newaxis, ...],
            "prior_draws": np.array(
                np.zeros((1, num_params), dtype=np.float32)
            ),  # prove that we're not accidentally leaking parameter info
            "sim_non_batchable_context": num_obs,
        }
    )

    posterior_samples_y = trainer.amortizer.sample(
        y_obs_configured, n_samples=num_ppred_samples
    )  # posterior_samples_y ~ q_φ(θ|y), shape: (batch_size, num_ppred,)

    _, parameter_dim = posterior_samples_y.shape

    conditional_posterior_samples = np.empty(
        (num_ppred_samples, num_posterior_samples, parameter_dim)
    )

    # ppred_sample ~ p(y'|y) = ∫q_φ(θ|y)p(y'|θ)dθ, shape: (num_obs, data_dim)
    ppred_sample = ppred_simulator(posterior_samples_y, n_obs=num_obs)

    y_obs_stacked = np.tile(y_obs, (num_ppred_samples, 1, 1))

    concatenated_y_ppred_configured = trainer.configurator(
        {
            "sim_data": np.concatenate([y_obs_stacked, ppred_sample], axis=1),
            "prior_draws": np.array(
                np.zeros((num_ppred_samples, num_params), dtype=np.float32)
            ),  # prove that we're not accidentally leaking parameter info
            "sim_non_batchable_context": num_obs * 2,
        }
    )

    conditional_posterior_samples = trainer.amortizer.sample(
        concatenated_y_ppred_configured, n_samples=num_posterior_samples
    )

    return posterior_samples_y, conditional_posterior_samples
