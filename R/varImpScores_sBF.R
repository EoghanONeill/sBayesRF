
#' @title Variable importances as defined by Hernandez et al. (2018)
#'
#' @description This measure defines the importance of a variable as the model-probability weighted sum of the number of splits on the variable of interest, divided by the sum over all variables of such weighted counts of splits.
#' @param object A sBF object.
#' @export
#' @return A vector of variable importances. The variables are ordered in the same order that they occur in columns of the input covariate matrix used to obtain the input BART-IS object.
#' @examples
#' #set the seed
#' set.seed(100)
#' #simulate some data
#'
#' #Train the object
#' #Obtain the variable importances
#'
varImpScores_sBF<-function(object){
  #object will be bartBMA object.
  imp_vars2=get_weighted_var_imp(num_vars=object$numvars,BIC=object$model_probs,sum_trees=object$tree_tables)
  res<-apply(imp_vars2[[3]],2,sum)
  #create varImpPlot command
  vIP<-rep(NA,length(res))
  total_weighted_var_counts<-sum(res)
  #now get variable inclusion probabilities
  vIP<-res/total_weighted_var_counts
  class(vIP)<-"varImpScores.sBF"
  vIP
}
