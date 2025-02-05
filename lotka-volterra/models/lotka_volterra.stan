functions {
  vector dz_dt(real t,       // time
               vector z,     // system state {prey, predator}
               array[] real theta // parameters
               //real[] x_r,   // unused data
               //int[] x_i
               ) {
    real u = z[1];
    real v = z[2];

    real alpha = theta[1];
    real beta = theta[2];
    real gamma = theta[3];
    real delta = theta[4];

    real du_dt = (alpha - beta * v) * u;
    real dv_dt = (-gamma + delta * u) * v;

    return to_vector({du_dt, dv_dt});
  }
}

data {
  int<lower = 0> N;          // number of measurement times exl. year 0
  array[N] real ts;                // measurement times > 0
  array[2] real y_init;            // initial measured populations
  array[N,2] real <lower = 0> y;   // measured populations
}
parameters {
  array[4] real<lower = 0> theta;   // { alpha, beta, gamma, delta }
  vector <lower = 0> [2] z_init;  // initial population
  array[2] real<lower = 0> sigma;   // measurement errors
}
transformed parameters {
  array[N] vector[2] z
    = ode_rk45_tol(dz_dt, z_init, 1.0, ts, 1e-6, 1e-5, 1000, theta);
}
model {
  theta[{1, 3}] ~ normal(1, 0.5);
  theta[{2, 4}] ~ normal(0.05, 0.05);
  sigma ~ lognormal(-1, 1);
  z_init ~ lognormal(log(10), 1);
  for (k in 1:2) {
    y_init[k] ~ lognormal(log(z_init[k]), sigma[k]);
    y[ , k] ~ lognormal(log(z[, k]), sigma[k]);
  }
}

generated quantities {
  real loglik = 0;
  vector[N + 1] log_lik;
  log_lik[1] = 0;
  for (k in 1:2) {
    log_lik[1] += lognormal_lpdf(y_init[k]|log(z_init[k]), sigma[k]);
    loglik += lognormal_lpdf(y_init[k]|log(z_init[k]), sigma[k]);
    loglik += lognormal_lpdf(y[ , k]|log(z[, k]), sigma[k]);
    }
  for (n in 1:N) {
    log_lik[n + 1] = 0;
    for (k in 1:2) {
      log_lik[n + 1] += lognormal_lpdf(y[n, k]|log(z[n, k]), sigma[k]);
      }
      }
}
