# Case study: Lotka-Volterra model

Here are the source files for repeating the experiment of calibration assessment for the
Lotka-Volterra predator-prey ODE model presented in our paper.

The scripts for running the experiment are in [lv-prior-sbc-pth-init.R](lv-prior-sbc-pth-init.R) and
[lv-posterior-sbc-pth-init.R](lv-posterior-sbc-pth-init.R) for the experiments presented in the
paper. Additionally, [lv-prior-sbc.R](lv-prior-sbc.R) and [lv-posterior-sbc.R](lv-posterior-sbc.R)
create the experiments without using Pathfinder for initializing the MCMC chains. These two
experiments are excluded from the paper, but included for comparison here.

The results and visualizations presented in the paper are summarized and briefly discussed in
[lotka-volterra-sbc.qmd](lotka-volterra-sbc.qmd), available in
[pdf](lotka-volterra-sbc.pdf) and [html](lotka-volterra-sbc.html).

Additionally, a slightly more in-depth look at the multi-modal posteriors in posterior SBC is
provided in [lotka_volterra_posterior_sbc.qmd](lotka_volterra_posterior_sbc.qmd)
([pdf](lotka_volterra_posterior_sbc.pdf) / [html](lotka_volterra_posterior_sbc.html)).


[lotka_volterra_prior_sbc.qmd](lotka_volterra_prior_sbc.qmd) ([pdf](lotka_volterra_prior_sbc.pdf) /
[html](lotka_volterra_prior_sbc.html)) shows some prior predictive draws and trajectories of the
population dynamics in addition to the results of prior SBC without Pathfinder initialization..

The Stan implementations for the models are available in the models folder.