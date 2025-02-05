data {
    int <lower = 0> J; // Number of groups
    int <lower = 0> N; // Number of samples per group
    matrix[N, J] y;
}
parameters {
    vector[J] theta_tilde;
    real mu; // population mean
    real<lower=0> tau; // hyper-parameter of sdv
    real<lower=0> sigma;
}
transformed parameters {
    vector[J] theta;
    for (j in 1:J) {
        theta[j] = mu + tau * theta_tilde[j]; // mean of group j
    }
}
model {
    tau ~ normal(0, 1);
    mu ~ normal(0, 1);
    theta_tilde ~ normal(0, 1);
    sigma ~ normal(0, 1);

    for (n in 1:N) {
        y[n, ] ~ normal(theta, sigma);
    }
}

generated quantities {
    real loglik = 0;
    for (n in 1:N) {
        loglik += normal_lpdf(y[n,]| theta, sigma);
    }
}
