import bayesflow as bf


def get_amortizer(cfg, num_params):
    summary_net = bf.networks.SetTransformer(**cfg.summary_net_args)

    inference_net_settings = cfg.inference_net_args.to_dict()
    inference_net_settings["num_params"] = num_params
    inference_net = bf.networks.InvertibleNetwork(**inference_net_settings)

    amortizer = bf.amortizers.AmortizedPosterior(
        summary_net=summary_net,
        inference_net=inference_net,
    )

    return amortizer
