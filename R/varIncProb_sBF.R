#' @title Variable inclusion probabilities as defined by Linero (2018)
#'
#' @description This measure defines the posterior inclusion probability of a variable as the model-probability weighted sum of indicator variables for whether the variable was used in any splitting rules.
#' @param object A sBF object obtained using the sBayesRF_func function.
#' @export
#' @return A vector of posterior inclusion probabilities. The variables are ordered in the same order that they occur in columns of the input covariate matrix used to obtain the input sBF object.
#' @examples
#' #set the seed
#' set.seed(100)
#'
#' #Obtain the variable importances
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
#'varIncProb_sBF(test_sbf)


varIncProb_sBF <- function(object){
  #object will be bartBMA object.
  imp_vars2=get_weighted_var_imp(num_vars=object$numvars,BIC=object$model_probs,sum_trees=object$tree_tables)
  #res<-apply((imp_vars2[[3]]>0)*imp_vars2[[1]],2,sum)
  res<- t(imp_vars2[[2]]>0)%*%imp_vars2[[1]]

  #create varImpPlot command
  class(res)<-"varIncProb.sBF"
  res
}
