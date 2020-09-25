#' @title Parallel Safe-Bayesian Random Forest
#'
#' @description A parallelized implementation of the Safe-Bayesian Random Forest described by Quadrianto and Ghahramani (2015)
#' @param lambda A real number between 0 and 1 that determines the splitting probability in the prior (which is used as the importance sampler of tree models) (relevant if tree_prior==1 or imp_sampler==1). Quadrianto and Ghahramani (2015) recommend a value less than 0.5 .
#' @param num_trees The number of trees to be sampled.
#' @param seed The seed for random number generation.
#' @param num_cats The number of possible values for the outcome variable.
#' @param y The training data vector of outcomes. This must be a (numeric) vector of integers between 1 and num_cats.
#' @param original_datamat The original training data (excluding the outcome  variable). Currently all variables must be continuous. The training data does not need to be transformed before being entered to this function.
#' @param alpha_parameters Vector of prior parameters. Dirichlet prior for class probabilities used in Dirichlet-multinomial model for each terminal node.
#' @param beta_par The power to which the likelihood is to be raised. For BMA, set beta_par=1. Recommended value is between 0 and 1.
#' @param test_datamat The test data. This matrix must have the same number of columns (variables) as the training data. Currently all variables must be continuous. The test data does not need to be transformed before being entered to this function.
#' @param valid_trees If equal to 1, restrict splits so that they describe feasible/valid partitions. e.g. can't have a rule x1<0.75 as the splitting rule for the left child node of a parent node with the splitting rule x1<0.5
#' @param tree_prior 1 = BART prior, 2= spike-and-tree, otherwise default prior by Novi and Quandrianto
#' @param imp_sampler Importance sampler for trees. 1 = BART prior, 2= spike-and-tree, otherwise default prior by Novi and Quandrianto
#' @param alpha_BART The alpha parameter for the standard BART prior.
#' @param beta_BART The beta parameter for the standard BART prior.
#' @param s_t_hyperprior If equals 1 and spike_tree equals 1, then a beta distribution hyperprior is placed on the variable inclusion probabilities for the spike and tree prior. The hyperprior parameters are a_s_t and b_s_t.
#' @param p_s_t If tree_prior=2 and s_t_hyperprior=0, then p_s_t is the prior variable inclusion probability.
#' @param a_s_t If tree_prior=2 and s_t_hyperprior=1, then a_s_t is a parameter of a beta distribution hyperprior.
#' @param b_s_t If tree_prior=2 and s_t_hyperprior=1, then b_s_t is a parameter of a beta distribution hyperprior.
#' @param lambda_poisson This is a parameter for the Spike-and-Tree prior. It is the parameter for the (truncated and conditional on the number of splitting variables) Poisson prior on the number of terminal nodes.
#' @param in_samp_preds If equal to 1, output the in-sample predicted probabilities for the training data.
#' @param save_tree_tables If equal to 1, output includes a list of tables describing the structures of all trees in the forest. Require this for out-of-sample predictions on a new dataset, and for variable importance measures.
#' @param find_power If equal to 1, instead of using beta_par, the power for the likelihood weight will be found by searching over n values from 1/n, 2/n,..., to 1, where n is the training sample size. The criterion is insample accuracy (proportion of correct predictions in-sample).
#' @param ncores The number of cores to be used in parallelization.
#' @return The output is a list containing:
#' \item{orig_logliks}{A vector of unormalized marginal likelihoods for each model.}
#' \item{model_probs}{A vector of model weights. The likelihoods are raised to the power beta_par and normalized to sum to 1.}
#' \item{tree_tables}{A list of tables (matrices) describing the tree structures,
#' The first (second) column gives the row number of the left (right) daughter node,
#' the third column gives the splitting variable,
#' the fourth column gives the splitting point (after a PIT ECDF transformation),
#' the fifth column indicates if the node is a terminal node,
#' the remaining columns give category probabilities for the individual nodes.}
#' \item{cat_probs}{A matrix of predicted test data probabilities, with columns corresponding to categories and rows corresponding to observations.}
#' \item{cat_probs_insamp}{A matrix of predicted in-sample (training) data probabilities, with columns corresponding to categories and rows corresponding to observations.}
#' \item{pred_categories}{Vector of categories with the highest probability for each test data observation. In the case of equal probabilities for 2 or more categories, the selected category is the lowest numbered category.}
#' \item{pred_categories_insamp}{Vector of categories with the highest probability for each training data observation. In the case of equal probabilities for 2 or more categories, the selected category is the lowest numbered category.}
#' \item{power_operating_curve}{A vector of in-sample accuracies for different possible values of the power parameter beta_par. The n values are 1/n, 2/n, up to 1, where n is the training sample size.}
#' \item{selected_power}{The selected power for the likelihood that maximizes in-sample accuracy.}
#' \item{numvars}{Number of covariates/variables}
#' \item{original_datamat}{The traininfg data covariate matrix.}
#' \item{y_original}{The training data outcome vector.}
#' \item{num_cats}{Number of possible categories (inital input parameter)}
#' \item{set_power}{The beta_par value used as an input parameter.}
#' @return A matrix of probabilities with the number of rows equl to the number of test observations and the number of columns equal to the number of possible outcome categories.
#' @useDynLib sBayesRF, .registration = TRUE
#' @examples
#'
#' Num_vars <- 50
#' Num_obs <- 1000
#' Num_cats <- 2
#'
#' alpha_parameters <- rep(1,Num_cats)
#' beta_par <- 1
#'
#' #generate training data
#' x1 = c(rnorm(Num_obs/2, 0,1), rnorm(Num_obs/2,3,1))
#' x2 = rnorm(Num_obs)
#' x3 = rnorm(Num_obs)
#' class= factor(rep(1:2, each=Num_obs/2))
#' data_original1 <- cbind(x1,x2,x3)
#' y <- as.numeric(class)
#'
#' #generate test data
#' x1test = c(rnorm(Num_obs/2, 0,1), rnorm(Num_obs/2,3,1))
#' x2test = rnorm(Num_obs)
#' x3test = rnorm(Num_obs)
#' classtest= factor(rep(1:2, each=Num_obs/2))
#'
#' data_test1 <- cbind(x1test,x2test,x3test)
#' ytest <- as.numeric(classtest)
#'
#'
#' lambda <- 0.45
#' Num_trees <- 2000
#'
#' seed1 <- 42
#' ncores <- 7
#'
#' test_sbf <- sBayesRF_func(lambda, Num_trees,
#'                                 seed1, Num_cats,
#'                                 y, data_original1,
#'                                 alpha_parameters, beta_par,
#'                                 data_test1,
#'                                 ncores=ncores,
#'                                 tree_prior=1,
#'                                 imp_sampler=1,
#'                                 find_power = 0)
#'
#' @export

sBayesRF_func <- function(lambda=0.45,
                          num_trees=1000,
                          seed,
                          num_cats,
                          y,
                          original_datamat,
                          alpha_parameters=rep(1,num_cats),
                          beta_par=1,
                          test_datamat=matrix(0.0,0,0),
                          ncores=1,
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
                          in_samp_preds=1,
                          save_tree_tables=1,
                          find_power=0){

  if(is.vector(original_datamat) | is.factor(original_datamat)| is.data.frame(original_datamat)) original_datamat = as.matrix(original_datamat)
  if(is.vector(test_datamat) | is.factor(test_datamat)| is.data.frame(test_datamat)) test_datamat = as.matrix(test_datamat)

  if((!is.matrix(original_datamat))) stop("argument x.train must be a double matrix")
  if((!is.matrix(test_datamat)) ) stop("argument x.test must be a double matrix")

  if(nrow(original_datamat) != length(y)) stop("number of rows in x.train must equal length of y.train")

  if(nrow(test_datamat) >0){
    if((ncol(test_datamat)!=ncol(original_datamat))) stop("input x.test must have the same number of columns as x.train")
  }




  #any(sort(unique(Y_all_training)) != min(Y_all_training):(min(Y_all_training)+length(unique(Y_all_training))-1) )
  #any(sort(unique(Y_all_training))!= min(Y_all_training):(min(Y_all_training)+length(unique(Y_all_training))-1) )

if(any(sort(unique(y))!= min(y):(min(y)+length(unique(y))-1) ) ){
  stop("Possible classes must be consecutive integers")
}


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
                                    save_tree_tables,
                                    find_power)



  is_test_data=0
  if(nrow(test_datamat)>0){
    is_test_data=1
  }


  if(find_power==1){


    if(is_test_data==0 & in_samp_preds==0){
      if(save_tree_tables==1){

        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "tree_tables",
                            "power_operating_curve",
                            "selected_power");


      }else{
        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "power_operating_curve",
                            "selected_power");
      }
    }

    if(is_test_data==1 & in_samp_preds==0){

      if(save_tree_tables==1){
        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs",
                            "tree_tables",
                            "pred_categories",
                            "power_operating_curve",
                            "selected_power");

      }else{
        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs",
                            "pred_categories",
                            "power_operating_curve",
                            "selected_power");

      }
    }

    if(is_test_data==0 & in_samp_preds==1){

      if(save_tree_tables==1){

        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                           "cat_probs_insamp",
                           "tree_tables",
                           "pred_categories_insamp",
                           "power_operating_curve",
                           "selected_power");

      }else{
        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs_insamp",
                            "pred_categories_insamp",
                            "power_operating_curve",
                            "selected_power");

      }
    }
    if(is_test_data==1 & in_samp_preds==1){
      if(save_tree_tables==1){

        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs",
                            "cat_probs_insamp",
                            "tree_tables",
                            "pred_categories",
                            "pred_categories_insamp",
                            "power_operating_curve",
                            "selected_power");

      }else{

        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs",
                            "cat_probs_insamp",
                            "pred_categories",
                            "pred_categories_insamp",
                            "power_operating_curve",
                            "selected_power");

      }
    }


  }else{

    if(is_test_data==0 & in_samp_preds==0){
      if(save_tree_tables==1){

        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "tree_tables");


      }else{
        names(sBRFcall) = c("orig_logliks",
                            "model_probs");
      }
    }

    if(is_test_data==1 & in_samp_preds==0){

      if(save_tree_tables==1){
        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs",
                            "tree_tables",
                            "pred_categories");

      }else{
        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs",
                            "pred_categories");

      }
    }

    if(is_test_data==0 & in_samp_preds==1){

      if(save_tree_tables==1){

        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs_insamp",
                            "tree_tables",
                            "pred_categories_insamp");

      }else{
        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs_insamp",
                            "pred_categories_insamp");

      }
    }
    if(is_test_data==1 & in_samp_preds==1){
      if(save_tree_tables==1){

        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs",
                            "cat_probs_insamp",
                            "tree_tables",
                            "pred_categories",
                            "pred_categories_insamp");

      }else{

        names(sBRFcall) = c("orig_logliks",
                            "model_probs",
                            "cat_probs",
                            "cat_probs_insamp",
                            "pred_categories",
                            "pred_categories_insamp");

      }
    }
    sBRFcall$set_power = beta_par

}

  sBRFcall$numvars = ncol(original_datamat)
  sBRFcall$original_datamat = original_datamat
  sBRFcall$y_original = y
  sBRFcall$num_cats = num_cats

  class(sBRFcall)<-"sBF"


  sBRFcall
}
