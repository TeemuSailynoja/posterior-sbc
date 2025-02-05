#!/usr/bin/env Rscript
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
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
set.seed(seed)

library("SBC")
library("cmdstanr")
library("posterior")

# Function for generating prior predictive draws
prior_predict <- function(n_groups = 50,
                          n_obs_pg = 5,
                          qtau = NULL,
                          qsigma = NULL,
                          seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  mu <- 0
  tau <- qnorm(.5 + .5 * qtau, 0, 1)
  theta <- rnorm(n_groups, mu, tau)
  sigma <- qnorm(.5 + .5 * qsigma, 0, 1)
  y <- matrix(rnorm(n_obs_pg * n_groups, theta, sigma),
    ncol = n_groups,
    byrow = TRUE
  )
  list(
    variables = list(
      mu = mu,
      tau = tau,
      sigma = sigma,
      loglik = sum(apply(y, 1, dnorm, theta, sigma, log = TRUE))
    ),
    generated = list(N = n_obs_pg, J = n_groups, y = y)
  )
}

# Generate two observations the results in the paper use 0.05, and 0.95
# the posterior SBC results with 1000 iterations use 0.1 and 0.9.
observations <- list(
  weak_likelihood = prior_predict(qtau = .05,
                                  qsigma = .95,
                                  seed = seed)$generated,
  strong_likelihood = prior_predict(qtau = .95,
                                    qsigma = .05,
                                    seed = seed)$generated
)

# Load in the models for posterior predictive sampling.
models <- list(
  cp = cmdstan_model("models/schools_cp_general_yrep.stan"),
  ncp = cmdstan_model("models/schools_ncp_general_yrep.stan")
)

# Fit both models to both observations
fits <- lapply(models, \(m) {
  lapply(observations, \(o) {
    m$sample(o,
      parallel_chains = 4,
      refresh = 10 * n_sbc_iter / 4,
      chains = 4,
      adapt_delta = .99,
      max_treedepth = 12,
      iter_warmup = 1000,
      iter_sampling = 10 * n_sbc_iter / 4 # Have enough draws after thinning.
    )
  })
})

# Generate datasets for posterior SBC by joining the observation and
# a posterior predictive draw.
sbc_datasets <- lapply(names(models), \(m) {
  data_sets <- lapply(names(observations), \(o) {
    fit <- fits[[m]][[o]]
    SBC:::new_SBC_datasets(
      variables = thin_draws(merge_chains(fit$draws(
        c("mu", "tau", "theta", "sigma", "loglik"),
        format = "matrix"
      )), 10),
      generated = lapply(1:n_sbc_iter, \(i) {
        list(
          N = 10,
          J = 50,
          y = rbind(
            observations[[o]]$y,
            matrix(thin_draws(merge_chains(fit$draws("yrep", format = "matrix")), 10)[i, ],
              ncol = ncol(observations[[o]]$y)
            )
          )
        )
      })
    )
  })
  names(data_sets) <- names(observations)
  data_sets
})

names(sbc_datasets) <- names(models)

# Backends for the SBC package. These define the parameters for MCMC used
# in running SBC.
backends <- list(
  cp = SBC_backend_cmdstan_sample(
    cmdstan_model("models/schools_cp_general.stan"),
    parallel_chains = 4,
    chains = 4,
    adapt_delta = .99,
    refresh = 10 * n_sbc_iter / 4,
    max_treedepth = 12,
    iter_warmup = 1000,
    iter_sampling = 10 * n_sbc_iter / 4
  ),
  ncp = SBC_backend_cmdstan_sample(
    cmdstan_model("models/schools_ncp_general.stan"),
    parallel_chains = 4,
    chains = 4,
    adapt_delta = .99,
    refresh = 10 * n_sbc_iter / 4,
    max_treedepth = 12,
    iter_warmup = 1000,
    iter_sampling = 10 * n_sbc_iter / 4
  )
)

# Finally, run posterior SBC and save results.
sbc_results <- lapply(names(models), \(m) {
  lapply(names(observations), \(o) {
    print(paste(m, o))
    res <- compute_SBC(
      sbc_datasets[[m]][[o]],
      backends[[m]],
      keep_fits = FALSE,
      thin_ranks = 10,
      cache_mode = "results",
      cache_location = paste(
        "./sbc_results/posterior_sbc",
        m,
        o,
        "niter",
        n_sbc_iter,
        "seed",
        seed,
        sep = "-"
      )
    )
    if (interactive()) {
      p <- plot_ecdf_diff(res, c("loglik", "mu", "tau", "sigma"))
      print(p)
    }
    res
  })
}) |> unlist(recursive = FALSE)

# Make quick plots of PIT ECDF difference and parameter recoveryfor the runs.
names(sbc_results) <-
  sapply(c("cp", "ncp"),
         \(m) paste(m, c("weak_likelihood", "strong_likelihood"), sep = "-"))

sbc_results |>
  seq_along() |>
  sapply(\(i) {
    p1 <- plot_ecdf_diff(sbc_results[[i]],
                         c("loglik", "mu", "tau", "sigma"))
    p2 <- plot_sim_estimated(sbc_results[[i]],
                             c("loglik", "mu", "tau", "sigma"))

    ggplot2::ggsave(paste("./sbc_results/posterior_sbc_ecdf_plot_",
                          names(sbc_results)[i],
                          "-niter_",
                          n_sbc_iter,
                          "-seed_",
                          seed,
                          ".png",
                          sep = ""),
                    p1)
    ggplot2::ggsave(paste("./sbc_results/posterior_sbc_sim_estim_plot_",
                          names(sbc_results)[i],
                          "-niter_",
                          n_sbc_iter,
                          "-seed_",
                          seed,
                          ".png",
                          sep = ""),
                    p2)
  })
