data {
    int <lower = 0> J; // Number of groups
    int <lower = 0> N; // Number of samples per group
    matrix[N, J] y;
}
parameters {
    vector[J] theta; // mean of group j
    real mu; // population mean
    real<lower=0> tau; // hyper-parameter of sdv
    real<lower=0> sigma; // hyper-parameter of sdv
}
model {
    tau ~ normal(0, 1);
    mu ~ normal(0, 1);
    theta ~ normal(mu, tau);
    sigma ~ normal(0, 1);

    for (n in 1:N) {
        y[n, ] ~ normal(theta, sigma);
    }
}
generated quantities {
    array[N, J] real yrep;
    real loglik = 0;
    for (n in 1:N) {
        yrep[n, ] = normal_rng(theta, sigma);
        loglik += normal_lpdf(y[n, ] | theta, sigma);
        loglik += normal_lpdf(yrep[n, ]| theta, sigma);
    }
}
