#!/usr/bin/env Rscript
# This script recreates the experiment of running prior SBC
# on hte Lotka-Volterra model using Pathfinder initialization.
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
library("SBC")

set.seed(seed)

n_years <- 21
n_chains <- 4
n_samples <- 10 * n_sbc_iter / n_chains
iter_warmup <- 1000

lv_model <- cmdstanr::cmdstan_model("./models/lotka_volterra.stan")

prior_predictive_generator_years <- function(n_years) {
  # Some prior predictive samples produce negative populations.
  # We reject those, but out of curiosity I'll add a counter to keep
  # track of the number of rejected samples.
  valid_populations <- FALSE
  n_tries <- 0
  while (!valid_populations) {
    n_tries <- n_tries + 1
    # Draw parameter values from the prior:
    alpha <- qnorm(runif(1, pnorm(0, 1, .5), 1), 1, .5)
    beta <- qnorm(runif(1, pnorm(0, .05, .05), 1), .05, .05)
    gamma <- qnorm(runif(1, pnorm(0, 1, .5), 1), 1, .5)
    delta <- qnorm(runif(1, pnorm(0, .05, .05), 1), .05, .05)

    l0 <- rlnorm(1, log(10), 1)
    h0 <- rlnorm(1, log(10), 1)

    # Solve the ODE system to get populations for years 2-21.
    pops <- ode(
      f = function(h, l) {
        c((alpha - beta * l) * h, (-gamma + delta * h) * l)
      },
      var = c(h = h0, l = l0),
      times = seq(1, n_years, .1)
    )
    if (all(pops > 0)) {
      valid_populations <- TRUE
    }
  }

  # With valid populations, draw observation noise from priors and
  # simulate observations.
  sigma_h <- rlnorm(1, -1, 1)
  sigma_l <- rlnorm(1, -1, 1)
  hare_pelts <- rlnorm(n_years, log(pops[seq(1, 201, 10), 1]), sigma_h)
  lynx_pelts <- rlnorm(n_years, log(pops[seq(1, 201, 10), 2]), sigma_l)

  # Return the ground truth as `variables` and observations as `generated`
  list(
    variables = list(
      theta = c(alpha, beta, gamma, delta),
      z_init = c(h0, l0),
      sigma = c(sigma_h, sigma_l),
      loglik = sum(dlnorm(hare_pelts,
                          log(pops[seq(1, 201, 10), 1]),
                          sigma_h,
                          log = TRUE)) +
        sum(dlnorm(lynx_pelts,
                   log(pops[seq(1, 201, 10), 2]),
                   sigma_l,
                   log = TRUE))
    ),
    generated = list(
      N = n_years - 1,
      ts = 2:n_years,
      y_init = c(hare_pelts[1], lynx_pelts[1]),
      y = matrix(c(hare_pelts[-1], lynx_pelts[-1]), ncol = 2),
      n_tries = n_tries
    )
  )
}

prior_predictive_generator <- SBC_generator_function(
  prior_predictive_generator_years,
  n_years = n_years
)

print("Generating prior predictive datasets")
prior_predictive_datasets <- generate_datasets(
  prior_predictive_generator,
  n_sims = n_sbc_iter
)
print("Generated prior predictive datasets.")

print("Running prior SBC")
stan_backend <- SBC_backend_cmdstan_sample(
  lv_model,
  iter_warmup = iter_warmup,
  iter_sampling = n_samples,
  parallel_chains = 4,
  chains = 4,
  refresh = 0,
  init_factory = \(stan_data) {
    cmdstanr::cmdstan_model("./models/lotka_volterra.stan")$pathfinder(
      data = stan_data,
      num_paths = 10,
      single_path_draws = 40,
      draws = 400,
      history_size = 9,
      max_lbfgs_iters = 100,
      refresh = 0,
      seed = as.integer(stan_data$y_init[1] + stan_data$y_init[2])
    )
  }
)

results <- compute_SBC(
  prior_predictive_datasets,
  stan_backend,
  cache_mode = "results",
  cache_location = paste(
    "./sbc_results/results-prior_sbc-pth-n_iter_", n_sbc_iter,
    "seed_", seed,
    sep = ""
  ),
  keep_fits = FALSE,
  thin_ranks = 10
)
