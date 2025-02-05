import numpy as np
from numba import njit


def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------

    Arguments:
    batch_size : int -- the number of samples to draw from the prior
    ----------

    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
    """

    # Prior ranges for the simulator
    # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 4.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # mu_tau_e ~ U(0.05, 0.6)
    # tau_m ~ U(0.06, 0.8)
    # sigma ~ U(0, 0.3)
    # varsigma ~ U(0, 0.3)
    # lambda ~ U(.5,4)
    n_parameters = 8
    p_samples = np.random.uniform(
        low=(0.1, 0.5, 0.1, 0.05, 0.06, 0.0, 0.0, 0.5),
        high=(3.0, 3.0, 0.9, 0.6, 0.8, 0.3, 0.3, 4.0),
        size=(batch_size, n_parameters),
    )
    return p_samples.astype(np.float32)


@njit
def diffusion_trial(
    drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, lam, dc=1.0, dt=0.005
):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.0
    evidence = boundary * beta

    # fixed parameter for collapsing
    # lamd is free paramter
    k = 3
    delt = -1

    # Simulate a single DM path
    while evidence > (1 - np.exp(-((n_steps * dt / lam) ** k))) * (
        -0.5 * delt * boundary
    ) and evidence < (
        boundary - (1 - np.exp(-((n_steps * dt / lam) ** k))) * (-0.5 * delt * boundary)
    ):
        # DDM equation
        evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt

    # visual encoding time for each trial
    tau_e_trial = np.random.normal(mu_tau_e, varsigma)

    # N200 latency
    z = np.random.normal(tau_e_trial, sigma)

    if evidence >= (
        boundary - (1 - np.exp(-((n_steps * dt / lam) ** k))) * (-0.5 * delt * boundary)
    ):
        choicert = tau_e_trial + rt + tau_m
    else:
        choicert = -tau_e_trial - rt - tau_m
    return choicert, z


@njit
def diffusion_condition(params, n_trials):
    """Simulates a diffusion process over an entire condition."""

    drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, lam = params
    choicert = np.empty(n_trials)
    z = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i], z[i] = diffusion_trial(
            drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, lam
        )
    return choicert, z


def batch_simulator(prior_samples, n_obs, dt=0.005, s=1.0):
    """
    Simulate multiple diffusion_model_datasets.
    """

    n_sim = prior_samples.shape[0]
    sim_choicert = np.empty((n_sim, n_obs), dtype=np.float32)
    sim_z = np.empty((n_sim, n_obs), dtype=np.float32)

    # Simulate diffusion data
    for i in range(n_sim):
        sim_choicert[i], sim_z[i] = diffusion_condition(prior_samples[i], n_obs)

    sim_data = np.stack([sim_choicert, sim_z], axis=-1)
    return sim_data
