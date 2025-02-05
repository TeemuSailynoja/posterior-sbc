#!/usr/bin/env Rscript
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  setwd(args[1])
  if (length(args) == 3) {
    n_sbc_iter <- as.integer(args[2])
    seed <- as.integer(args[3])
  } else {
    n_sbc_iter <- 500
    seed <- 8769
  }
  options(mc.cores = parallel::detectCores())
  # Enabling parallel processing via future
  library("future")
  plan(multisession)
} else {
  # Original experiment was done without parallel SBC iterations.
  n_sbc_iter <- 500
  seed <- 8769
}

library("SBC")
library("cmdstanr")

set.seed(seed)


generate_observation <- function(n_groups = 50,
                                 n_obs_pg = 5) {
  mu <- rnorm(1, 0, 1)
  tau <- abs(rnorm(1, 0, 1))
  theta <- rnorm(n_groups, mu, tau)
  sigma <- abs(rnorm(1, 0, 1))
  y <- matrix(rnorm(n_obs_pg * n_groups, theta, sigma),
              ncol = n_groups,
              byrow = TRUE)
  list(
    variables = list(
      mu = mu,
      tau = tau,
      #theta = theta,
      sigma = sigma,
      loglik = sum(apply(y, 1, dnorm, theta, sigma, log = TRUE))
    ),
    generated = list(N = n_obs_pg, J = n_groups, y = y)
  )
}

models <- list(
  cp = cmdstan_model("models/schools_cp_general.stan"),
  ncp = cmdstan_model("models/schools_ncp_general.stan")
)

generator <- SBC_generator_function(generate_observation)
sbc_datasets <- generate_datasets(generator, n_sbc_iter)

saveRDS(
  sbc_datasets,
  paste("./sbc_results/prior_sbc_datasets_seed_",
    seed,
    "n_iter",
    n_sbc_iter,
    ".rds",
    sep = ""
  )
)

backends <- models |> lapply(
  SBC_backend_cmdstan_sample,
  parallel_chains = 4,
  chains = 4,
  adapt_delta = .99,
  max_treedepth = 12,
  iter_warmup = 1000,
  iter_sampling = 10 * n_sbc_iter / 4
)

sbc_results <- names(models) |> sapply(
  \(model_name) {
    compute_SBC(
      sbc_datasets,
      backends[[model_name]],
      keep_fits = FALSE,
      thin_ranks = 10,
      cache_mode = "results",
      cache_location = paste(
        "./sbc_results/prior_sbc_",
        model_name,
        "_niter_",
        n_sbc_iter,
        "_seed_",
        seed,
        sep = ""
      )
    )
  },
  simplify = FALSE,
  USE.NAMES = TRUE
)

sbc_results |>
  seq_along() |>
  sapply(\(i) {
    p1 <- plot_ecdf_diff(sbc_results[[i]],
                         c("loglik", "mu", "tau", "sigma"))
    p2 <- plot_sim_estimated(sbc_results[[i]],
                             c("loglik", "mu", "tau", "sigma"))

    ggplot2::ggsave(paste("./sbc_results/prior_sbc_ecdf_plot_",
                          names(sbc_results)[i],
                          "-niter_",
                          n_sbc_iter,
                          "-seed_",
                          seed,
                          ".png",
                          sep = ""),
                    p1)
    ggplot2::ggsave(paste("./sbc_results/prior_sbc_sim_estim_plot_",
                          names(sbc_results)[i],
                          "-niter_",
                          n_sbc_iter,
                          "-seed_",
                          seed,
                          ".png",
                          sep = ""),
                    p2)
  })
