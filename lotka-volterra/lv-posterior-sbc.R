#!/usr/bin/env Rscript
# This script recreates the experiment of running posterior SBC
# on the Lotka-Volterra model without Pathfinder initialization.
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  print(args)
  setwd(args[1])
  if (length(args) == 3) {
    n_sbc_iter <- as.integer(args[2])
    seed <- as.integer(args[3])
  } else {
    n_sbc_iter <- 500
    seed <- 2024
  }
  options(mc.cores = parallel::detectCores())
  # Enabling parallel processing via future
  library("future")
  plan(multisession)
} else {
  n_sbc_iter <- 500
  seed <- 2024
}

library("calculus")
library("cmdstanr")
library("posterior")
library("SBC")

set.seed(seed)

n_chains <- 4
n_samples <- 10 * n_sbc_iter / n_chains
iter_warmup <- 1000
save_stan_fit <- TRUE
stan_fit_dir <- "fits"
cache_dir <- "sbc_results"

lynx_hare_df <-
  read.csv("hudson-bay-lynx-hare.csv", comment.char = "#")

model_pred <- cmdstanr::cmdstan_model("models/lotka_volterra_pred.stan")

# Obtain posterior approximation with NUTS
if (file.exists(file.path(stan_fit_dir, "fit0.RDS"))) {
  fit0 <- readRDS(file.path(stan_fit_dir, "fit0.RDS"))
} else {
  fit0 <- model_pred$sample(
    data = list(
      N = nrow(lynx_hare_df) - 1,
      ts = 2:nrow(lynx_hare_df),
      y_init = c(lynx_hare_df$Hare[1], lynx_hare_df$Lynx[1]),
      y = as.matrix(lynx_hare_df[-1, c(3, 2)]),
      n_pred = 0
    ),
    seed = seed,
    refresh = 0,
    parallel_chains = n_chains,
    chains = n_chains,
    iter_warmup = iter_warmup,
    iter_sampling = n_samples,
    show_messages = FALSE
  )
  if (save_stan_fit) {
    fit0$save_object(file.path(stan_fit_dir, "fit0.RDS"))
  }
}

# Collect posterior draws into format usable by the SBC package.
variables <- fit0$draws(
  variables = c("theta", "z_init", "z", "sigma", "loglik"),
  format = "matrix"
) |>
  merge_chains() |>
  thin_draws(10)

posterior_predictions <- fit0$draws(
  variables = c("y_init_rep", "y_rep"),
  format = "matrix"
) |>
  merge_chains() |>
  thin_draws(1)

posterior_datasets <- SBC_datasets(
  variables = variables[, c(
    "theta[1]",
    "theta[2]",
    "theta[3]",
    "theta[4]",
    "z_init[1]",
    "z_init[2]",
    "sigma[1]",
    "sigma[2]",
    "loglik"
  )],
  generated = apply(
    posterior_predictions,
    1,
    \(sample) {
      list(
        N = nrow(lynx_hare_df) - 1,
        ts = 2:nrow(lynx_hare_df),
        y_init2 = sample[1:2],
        y2 = matrix(sample[-c(1, 2)], ncol = 2),
        y_init = c(lynx_hare_df$Hare[1], lynx_hare_df$Lynx[1]),
        y = as.matrix(lynx_hare_df[-1, c(3, 2)])
      )
    }
  )
)

sbc_model <- cmdstanr::cmdstan_model("models/lotka_volterra_postSBC.stan")
# Pass settings for MCMC sampling to the SBC backend.
stan_backend_pth <- SBC_backend_cmdstan_sample(
  sbc_model,
  iter_warmup = iter_warmup,
  iter_sampling = n_samples,
  chains = n_chains,
  parallel_chains = n_chains,
  seed = seed,
  refresh = 0
)

print("Computing SBC")
results_post_sbc <- compute_SBC(
  posterior_datasets,
  stan_backend_pth,
  cache_mode = "results",
  cache_location = file.path(
    cache_dir,
    paste(
      "results-posterior_sbc-n_iter_",
      n_sbc_iter,
      "_seed_",
      seed,
      sep = ""
    )
  ),
  keep_fits = FALSE,
  thin_ranks = 10,
  chunk_size = 1
)
