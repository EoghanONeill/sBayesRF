#' @title Variable inclusion probabilities as defined by Linero (2018)
#'
#' @description This measure defines the posterior inclusion probability of a variable as the model-probability weighted sum of indicator variables for whether the variable was used in any splitting rules.
#' @param object A sBF object obtained using the train_BART_IS or train_BART_IS_no_output function.
#' @export
#' @return A vector of posterior inclusion probabilities. The variables are ordered in the same order that they occur in columns of the input covariate matrix used to obtain the input sBF object.
#' @examples
#' #set the seed
#' set.seed(100)
#' #simulate some data

#' #Train the object
#'
#' #Obtain the variable importances
#'

varIncProb_sBF <- function(object){
  #object will be bartBMA object.
  imp_vars2=get_weighted_var_imp(num_vars=object$numvars,BIC=object$model_probs,sum_trees=object$tree_tables)
  #res<-apply((imp_vars2[[3]]>0)*imp_vars2[[1]],2,sum)
  res<- t(imp_vars2[[2]]>0)%*%imp_vars2[[1]]

  #create varImpPlot command
  class(res)<-"varIncProb.sBF"
  res
}
