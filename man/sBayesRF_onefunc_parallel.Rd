% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{sBayesRF_onefunc_parallel}
\alias{sBayesRF_onefunc_parallel}
\title{Parallel Safe-Bayesian Random Forest}
\usage{
sBayesRF_onefunc_parallel(
  lambda,
  num_trees,
  seed,
  num_cats,
  y,
  original_datamat,
  alpha_parameters,
  beta_par,
  test_datamat,
  ncores
)
}
\arguments{
\item{lambda}{A real number between 0 and 1 that determines the splitting probability in the prior (which is used as the importance sampler of tree models). Quadrianto and Ghahramani (2015) recommend a value less than 0.5 .}

\item{num_trees}{The number of trees to be sampled.}

\item{seed}{The seed for random number generation.}

\item{num_cats}{The number of possible values for the outcome variable.}

\item{y}{The training data vector of outcomes. This must be a vector of integers between 1 and num_cats.}

\item{original_datamat}{The original test data. This matrix must have the same number of columns (variables) as the training data. Currently all variables must be continuous. The test data does not need to be transformed before being entered to this function.}

\item{alpha_parameters}{Vector of prior parameters.}

\item{beta_par}{The power to which the likelihood is to be raised. For BMA, set beta_par=1.}

\item{ncores}{The number of cores to be used in parallelization.}
}
\value{
A matrix of probabilities with the number of rows equl to the number of test observations and the number of columns equal to the number of possible outcome categories.
}
\description{
A parallelized implementation of the Safe-Bayesian Random Forest described by Quadrianto and Ghahramani (2015)
}
