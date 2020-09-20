#' @title Parallel Safe-Bayesian Random Forest
#'
#' @description A parallelized implementation of the Safe-Bayesian Random Forest described by Quadrianto and Ghahramani (2015)
#' @param lambda A real number between 0 and 1 that determines the splitting probability in the prior (which is used as the importance sampler of tree models). Quadrianto and Ghahramani (2015) recommend a value less than 0.5 .
#' @param num_trees The number of trees to be sampled.
#' @param seed The seed for random number generation.
#' @param num_cats The number of possible values for the outcome variable.
#' @param y The training data vector of outcomes. This must be a vector of integers between 1 and num_cats.
#' @param original_datamat The original training data. Currently all variables must be continuous. The training data does not need to be transformed before being entered to this function.
#' @param alpha_parameters Vector of prior parameters.
#' @param beta_par The power to which the likelihood is to be raised. For BMA, set beta_par=1.
#' @param original_datamat The original test data. This matrix must have the same number of columns (variables) as the training data. Currently all variables must be continuous. The test data does not need to be transformed before being entered to this function.
#' @param ncores The number of cores to be used in parallelization.
#' @return A matrix of probabilities with the number of rows equl to the number of test observations and the number of columns equal to the number of possible outcome categories.
#' @useDynLib sBayesRF, .registration = TRUE
#' @examples
#' Num_vars <- 50
#' Num_obs <- 100
#' Num_cats <- 5
#'
#' alpha_parameters <- rep(1,Num_cats)
#' beta_par <- 0.5
#'
#' data_original1 <- matrix( rnorm(Num_obs*Num_vars,mean=0,sd=1), Num_obs, Num_vars)
#' y <- sample(Num_cats,Num_obs, replace = TRUE)
#'
#' Num_test_vars <- 50
#' Num_test_obs <- 700
#'
#' data_test1 <- matrix( rnorm(Num_test_obs*Num_test_vars,mean=0,sd=1), Num_test_obs, Num_test_vars)
#'
#'
#' Num_split_vars <- 10
#'
#' lambda <- 0.45
#' Num_trees <- 100
#'
#' seed1 <- 42
#' ncores <- 1
#'
#' sBayesRF_parallel(lambda, Num_trees,
#'                   seed1, Num_cats,
#'                   y, data_original1,
#'                   alpha_parameters, beta_par,
#'                   data_test1,ncores)
#' @export

sBayesRF_more_priors <- function(lambda=0.45,num_trees=1000,
                              seed, num_cats,
                              y, original_datamat,
                              alpha_parameters=rep(1,num_cats),
                              beta_par=1,
                              test_datamat, ncores=1,
                              nu=3,
                              a=3,
                              sigquant=0.9,
                              valid_trees=1,
                              tree_prior=1,
                              imp_sampler=1,
                              alpha_BART=0.95,
                              beta_BART=2,
                              s_t_hyperprior=1,
                              p_s_t=0.5,
                              a_s_t=1,
                              b_s_t=3,
                              lambda_poisson=10,
                              l_quant=0.025,
                              u_quant=0.975,
                              in_samp_preds=1,
                              save_tree_tables=1){

  if(is.vector(original_datamat) | is.factor(original_datamat)| is.data.frame(original_datamat)) original_datamat = as.matrix(original_datamat)
  if(is.vector(test_datamat) | is.factor(test_datamat)| is.data.frame(test_datamat)) test_datamat = as.matrix(test_datamat)

  if((!is.matrix(original_datamat))) stop("argument x.train must be a double matrix")
  if((!is.matrix(test_datamat)) ) stop("argument x.test must be a double matrix")

  if(nrow(original_datamat) != length(y)) stop("number of rows in x.train must equal length of y.train")
  if((ncol(test_datamat)!=ncol(original_datamat))) stop("input x.test must have the same number of columns as x.train")

  sBRFcall=sBayesRF_more_priors_cpp(lambda,
                                    num_trees,
                                    seed,
                                    num_cats,
                                    y,
                                    original_datamat,
                                    alpha_parameters,
                                    beta_par,
                                    test_datamat,
                                    ncores,
                                    nu,
                                    a,#lambdaBART,
                                    valid_trees,
                                    tree_prior,
                                    imp_sampler,
                                    alpha_BART,
                                    beta_BART,
                                    s_t_hyperprior,
                                    p_s_t,
                                    a_s_t,
                                    b_s_t,
                                    lambda_poisson,
                                    in_samp_preds,
                                    save_tree_tables)



  is_test_data=0
  if(nrow(test_datamat)>0){
    is_test_data=1
  }

  if(is_test_data==0 & in_samp_preds==0){
    if(save_tree_tables==1){

      names(sBRFcall) = c("model_probs",
                          "tree_tables");


    }else{
      names(sBRFcall) = c("model_probs");
    }
  }

  if(is_test_data==1 & in_samp_preds==0){

    if(save_tree_tables==1){
      names(sBRFcall) = c("model_probs",
                          "cat_probs",
                          "tree_tables",
                          "pred_categories");

    }else{
      names(sBRFcall) = c("model_probs",
                          "cat_probs",
                          "pred_categories");

    }
  }

  if(is_test_data==0 & in_samp_preds==1){

    if(save_tree_tables==1){

      names(sBRFcall) = c("model_probs",
                         "cat_probs_insamp",
                         "tree_tables",
                         "pred_categories_insamp");

    }else{
      names(sBRFcall) = c("model_probs",
                          "cat_probs_insamp",
                          "pred_categories_insamp");

    }
  }
  if(is_test_data==1 & in_samp_preds==1){
    if(save_tree_tables==1){

      names(sBRFcall) = c("model_probs",
                          "cat_probs",
                          "cat_probs_insamp",
                          "tree_tables",
                          "pred_categories",
                          "pred_categories_insamp");

    }else{

      names(sBRFcall) = c("model_probs",
                          "cat_probs",
                          "cat_probs_insamp",
                          "pred_categories",
                          "pred_categories_insamp");

    }
  }

  sBRFcall$numvars = ncol(original_datamat)
  sBRFcall
}
