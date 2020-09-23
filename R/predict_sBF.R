
#' @title Classification predictions for test data from a trained safe-Bayesian Random Forest.
#'
#' @description This function produces prediction intervals for bart-is output.
#' @param object bartBMA object obtained from function bartis
#' @param newdata Test data for which predictions are to be produced. Default = NULL. If NULL, then produces prediction intervals for training data if no test data was used in producing the bartBMA object, or produces prediction intervals for the original test data if test data was used in producing the bartBMA object.
#' @param num_cores Number of cores used in parallel.
#' @param find_power If equal to 1, instead of using beta_par, the power for the likelihood weight will be found by searching over n values from 1/n, 2/n,..., to 1, where n is the training sample size. The criterion is insample accuracy (proportion of correct predictions in-sample).
#' @param beta_par The power to which the likelihood is to be raised. For BMA, set beta_par=1. Recommended value is between 0 and 1.
#' @export
#' @return The output is a list of length 2:
#' \item{PI}{A 3 by n matrix, where n is the number of observations. The first row gives the l_quant*100 quantiles. The second row gives the medians. The third row gives the u_quant*100 quantiles.}
#' \item{meanpreds}{An n by 1 matrix containing the estimated means.}
#'
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
#'                                 ncores=ncores,
#'                                 tree_prior=1,
#'                                 imp_sampler=1,
#'                                 find_power = 0)
#'
#'
#' test_pred_sbfobj <- predict_sBF(test_sbf,newdata = data_test1,
#'                                previous_power = 0,
#'                                find_power = 1)
#'

predict_sBF <-function(object,
                      newdata=NULL,
                      num_cores=1,
                      previous_power = 0,
                       find_power = 0,
                       beta_par=1){
  #object will be bartBMA object.


  if(previous_power ==1 & find_power==1){
    stop(
    "previous_power ==1 & find_power==1.
    Set one of these to 1 to indicate if
    should use power of likelihood applied when model
    was trained or should find a value for the likelihood power.
    Alternatively, set both to zero and set power using beta_par parameter.")
  }

  if(previous_power ==1 & find_power==0){

    if(is.null(object$selected_power)==TRUE){
      print("Using likelihood power from trained forest object. Power in trained object was user specified.")
      cat("\n")
      beta_par=object$set_power

    }else{
      print("Using likelihood power from trained forest object.
            Power in trained object was found by searching from 1/n, 2/n,... to 1 on training data.")
      cat("\n")
      beta_par=object$selected_power
    }
  }

  if(is.null(newdata) ){


    stop("Require test data (newdata)")



  }else{
    #if test data included in call to object
    ret<-pred_sBF(object$tree_tables,
                  object$model_probs,
                  object$original_datamat,
                  newdata,
                  object$orig_logliks,
                  object$y_original,
                  beta_par,
                  length(object$y_original),
                  nrow(newdata),
                  object$num_cats,
                  find_power,
                  ncores)

  }


  if(find_power==1){
    names(ret) <- c("orig_logliks",
                    "model_probs",
                    "cat_probs",
                    "pred_categories",
                    "power_operating_curve",
                    "selected_power")
  }else{
    names(ret) <- c("orig_logliks",
                    "model_probs",
                    "cat_probs",
                    "pred_categories")
  }


  class(ret)<-"preds.sBF"

  ret
}
