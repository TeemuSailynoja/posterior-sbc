# Posterior Simulation-Based Calibration Checking

This repository contains code for reproducing the three case studies presented in the paper 
**Posterior SBC: Simulation-Based Calibration Checking Conditional on Data** by 
Teemu Säilynoja, Marvin Schmitt, Paul Bürkner, and Aki Vehtari.

## Paper abstract
Simulation-based calibration checking (SBC) refers to the validation of an inference algorithm and
model implementation through repeated inference on data simulated from a generative model. In the
original and commonly used approach, the generative model uses parameters drawn from the prior, and
thus the approach is testing whether the inference works for simulated data generated with parameter
values plausible under that prior. This approach is natural and desirable when we want to test
whether the inference works for a wide range of datasets we might observe. However, after observing
data, we are interested in answering whether the inference works conditional on that particular
data. In this paper, we propose posterior SBC and demonstrate how it can be used to validate the
inference conditionally on observed data. We illustrate the utility of posterior SBC in three case
studies:
 (1) A simple multilevel model;
 (2) a model that is governed by differential equations; and
 (3) a joint integrative neuroscience model which is approximated via amortized Bayesian inference with neural networks.