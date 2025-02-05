from ml_collections import config_dict

cfg = config_dict.ConfigDict()

cfg.summary_net_args = dict(
    input_dim=3,
    summary_dim=32,
)


cfg.inference_net_args = {
    "coupling_design": "affine",
    "num_coupling_layers": 6,
}

cfg.epochs = 300
cfg.batch_size = 64
cfg.iterations_per_epoch = 1000
cfg.num_obs_min = 50
cfg.num_obs_max = 150
cfg.default_lr = 5e-4

cfg.num_test_datasets = 200
cfg.num_test_observations = 60
cfg.num_posterior_samples = 1000


cfg.param_names = {
    "m1a": [
        r"$\delta$",
        r"$\alpha$",
        r"$\beta$",
        r"$\mu_{(e)}$",
        r"$\tau_{(m)}$",
        r"$\sigma$",
        r"$s_{(\tau)}$",
    ],
    "m2": [
        r"$\delta$",
        r"$\alpha$",
        r"$\beta$",
        r"$\mu_{(e)}$",
        r"$\tau_{(m)}$",
        r"$\sigma$",
        r"$s_{(\tau)}$",
        r"$\gamma$",
    ],
    "m3": [
        r"$\delta$",
        r"$\alpha$",
        r"$\beta$",
        r"$\mu_{(e)}$",
        r"$\tau_{(m)}$",
        r"$\sigma$",
        r"$s_{(\tau)}$",
        r"$\theta_{(l)}$",
    ],
    "m4b": [
        r"$\delta$",
        r"$\alpha$",
        r"$\beta$",
        r"$\mu_{(e)}$",
        r"$\tau_{(m)}$",
        r"$\sigma$",
        r"$s_{(\tau)}$",
        r"$\theta_{(m)}$",
    ],
    "m5": [
        r"$\delta$",
        r"$\alpha$",
        r"$\beta$",
        r"$\mu_{(e)}$",
        r"$\tau_{(m)}$",
        r"$\sigma$",
        r"$s_{(\tau)}$",
        r"$\alpha_{\mathrm{slope}}$",
    ],
    "m6": [
        r"$\delta$",
        r"$\alpha$",
        r"$\beta$",
        r"$\mu_{(e)}$",
        r"$\tau_{(m)}$",
        r"$\sigma$",
        r"$s_{(\tau)}$",
        r"$\lambda$",
    ],
}
