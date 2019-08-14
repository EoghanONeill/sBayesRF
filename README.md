# sBayesRF

<!-- badges: start -->
<!-- badges: end -->

The goal of sBayesRF is to provide an implementation of the Safe-Bayesian Random Forest method described in:
Quadrianto, N., & Ghahramani, Z. (2014). A very simple safe-Bayesian random forest. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1297-1303.

## Installation

```r
library(devtools)
install_github("EoghanONeill/sBayesRF")
```

## Example


``` r
library(sBayesRF)

Num_vars <- 50
Num_obs <- 100
Num_cats <- 5

alpha_parameters <- rep(1,Num_cats)
beta_par <- 0.5

data_original1 <- matrix( rnorm(Num_obs*Num_vars,mean=0,sd=1), Num_obs, Num_vars)
y <- sample(Num_cats,Num_obs, replace = TRUE)

Num_test_vars <- 50
Num_test_obs <- 700

data_test1 <- matrix( rnorm(Num_test_obs*Num_test_vars,mean=0,sd=1), Num_test_obs, Num_test_vars)


Num_split_vars <- 10

lambda <- 0.45
Num_trees <- 100

seed1 <- 42
ncores <- 1

sBayesRF_parallel(lambda, Num_trees,
                          seed1, Num_cats,
                          y, data_original1,
                          alpha_parameters, beta_par,
                          data_test1,ncores)

```

