# include <RcppArmadillo.h>
// [[ Rcpp :: depends ( RcppArmadillo )]]
using namespace Rcpp;

//######################################################################################################################//


//' @title ECDF transformation of the training data
//'
//' @description Quadrianto and Ghahramani (2015) reccomend the use of the probability intergral transform to transform the continuous input features. The code is edited from https://github.com/dmbates/ecdfExample
//' @param originaldata Training data matrix
//' @export
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
NumericMatrix cpptrans_cdf(NumericMatrix originaldata){
  NumericMatrix transformedData(originaldata.nrow(), originaldata.ncol());
  for(int i=0; i<originaldata.ncol();i++){
    NumericVector samp= originaldata(_,i);
    NumericVector sv(clone(samp));
    std::sort(sv.begin(), sv.end());
    double nobs = samp.size();
    NumericVector ans(nobs);
    for (int k = 0; k < samp.size(); ++k)
      ans[k] = std::lower_bound(sv.begin(), sv.end(), samp[k]) - sv.begin();
    //NumericVector ansnum = ans;
    transformedData(_,i) = (ans+1)/nobs;
  }
  return transformedData;

}

//######################################################################################################################//

//' @title ECDF transformation of the test data
//'
//' @description Quadrianto and Ghahramani (2015) reccomend the use of the probability intergral transform to transform the continuous input features. The code is edited from https://github.com/dmbates/ecdfExample
//' @param originaldata Training data matrix
//' @param testdata Test data matrix
//' @export
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
NumericMatrix cpptrans_cdf_test(NumericMatrix originaldata, NumericMatrix testdata){
  NumericMatrix transformedData(testdata.nrow(), testdata.ncol());
  for(int i=0; i<testdata.ncol();i++){
    NumericVector samp= testdata(_,i);
    NumericVector svtest = originaldata(_,i);
    NumericVector sv(clone(svtest));
    std::sort(sv.begin(), sv.end());
    double nobs = samp.size();
    NumericVector ans(nobs);
    double nobsref = svtest.size();
    for (int k = 0; k < samp.size(); ++k){
      ans[k] = std::lower_bound(sv.begin(), sv.end(), samp[k]) - sv.begin();
    }
    //NumericVector ansnum = ans;
    transformedData(_,i) = (ans)/nobsref;
  }
  return transformedData;

}
//######################################################################################################################//

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

NumericVector find_term_nodes(NumericMatrix tree_table){
  arma::mat arma_tree(tree_table.begin(),tree_table.nrow(), tree_table.ncol(), false);

  //arma::vec colmat=arma_tree.col(4);
  //arma::uvec term_nodes=arma::find(colmat==-1);

  //arma::vec colmat=arma_tree.col(2);
  //arma::uvec term_nodes=arma::find(colmat==0);

  arma::vec colmat=arma_tree.col(4);
  arma::uvec term_nodes=arma::find(colmat==0);

  term_nodes=term_nodes+1;

  return(wrap(term_nodes));
}

//######################################################################################################################//

#include <math.h>       /* tgamma */
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

List get_treepreds(NumericVector original_y, int num_cats, NumericVector alpha_pars,
                   NumericMatrix originaldata, //NumericMatrix test_data,
                   NumericMatrix treetable//, NumericMatrix tree_data
) {
  // Function to make predictions from test data, given a single tree and the terminal node predictions, this function will be called
  //for each tree accepted in Occam's Window.

  //test_data is a nxp matrix with the same variable names as the training data the model was built on

  //tree_data is the tree table with the tree information i.e. split points and split variables and terminal node mean values

  //term_node_means is a vector storing the terminal node mean values
  arma::vec orig_y_arma= as<arma::vec>(original_y);
  arma::vec alpha_pars_arma= as<arma::vec>(alpha_pars);

  double lik_prod=1;
  double alph_prod=1;
  for(unsigned int i=0; i<alpha_pars_arma.n_elem;i++){
    alph_prod=alph_prod*tgamma(alpha_pars_arma(i));
  }
  double gam_alph_sum= tgamma(arma::sum(alpha_pars_arma));
  double alph_term=gam_alph_sum/alph_prod;

  arma::mat arma_tree_table(treetable.begin(), treetable.nrow(), treetable.ncol(), false);
  arma::mat arma_orig_data(originaldata.begin(), originaldata.nrow(), originaldata.ncol(), false);


  //arma::mat arma_tree(tree_data.begin(), tree_data.nrow(), tree_data.ncol(), false);
  //arma::mat testd(test_data.begin(), test_data.nrow(), test_data.ncol(), false);

  //NumericVector internal_nodes=find_internal_nodes_gs(tree_data);

  NumericVector terminal_nodes=find_term_nodes(treetable);
  //arma::vec arma_terminal_nodes=Rcpp::as<arma::vec>(terminal_nodes);
  //NumericVector tree_predictions;

  //now for each internal node find the observations that belong to the terminal nodes

  //NumericVector predictions(test_data.nrow());
  //List term_obs(terminal_nodes.size());

  if(terminal_nodes.size()==1){
    //double nodemean=tree_data(terminal_nodes[0]-1,5);				// let nodemean equal tree_data row terminal_nodes[i]^th row , 6th column. The minus 1 is because terminal nodes consists of indices starting at 1, but need indices to start at 0.
    //predictions=rep(nodemean,test_data.nrow());
    //Rcout << "Line 67 .\n";

    //IntegerVector temp_obsvec = seq_len(test_data.nrow())-1;
    //term_obs[0]= temp_obsvec;
    double denom_temp= orig_y_arma.n_elem+arma::sum(alpha_pars_arma);

    double num_prod=1;
    double num_sum=0;
    //Rcout << "Line 129.\n";

    for(int k=0; k<num_cats; k++){
      //assuming categories of y are from 1 to num_cats
      arma::uvec cat_inds= arma::find(orig_y_arma==k+1);
      double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
      arma_tree_table(0,5+k)= m_plus_alph/denom_temp ;

      //for likelihood calculation
      num_prod=num_prod*tgamma(m_plus_alph);
      num_sum=num_sum +m_plus_alph ;
    }

    lik_prod= alph_term*num_prod/tgamma(num_sum);

  }
  else{
    for(int i=0;i<terminal_nodes.size();i++){
      //arma::mat subdata=testd;
      int curr_term=terminal_nodes[i];

      int row_index;
      int term_node=terminal_nodes[i];
      //Rcout << "Line 152.\n";


      //WHAT IS THE PURPOSE OF THIS IF-STATEMENT?
      //Why should the ro index be different for a right daughter?
      //Why not just initialize row_index to any number not equal to 1 (e.g. 0)?
      row_index=0;

      // if(curr_term % 2==0){
      //   //term node is left daughter
      //   row_index=terminal_nodes[i];
      // }else{
      //   //term node is right daughter
      //   row_index=terminal_nodes[i]-1;
      // }




      //save the left and right node data into arma uvec

      //CHECK THAT THIS REFERS TO THE CORRECT COLUMNS
      //arma::vec left_nodes=arma_tree.col(0);
      //arma::vec right_nodes=arma_tree.col(1);

      arma::vec left_nodes=arma_tree_table.col(0);
      arma::vec right_nodes=arma_tree_table.col(1);



      arma::mat node_split_mat;
      node_split_mat.set_size(0,3);
      //Rcout << "Line 182. i = " << i << " .\n";

      while(row_index!=1){
        //for each terminal node work backwards and see if the parent node was a left or right node
        //append split info to a matrix
        int rd=0;
        arma::uvec parent_node=arma::find(left_nodes == term_node);

        if(parent_node.size()==0){
          parent_node=arma::find(right_nodes == term_node);
          rd=1;
        }

        //want to cout parent node and append to node_split_mat

        node_split_mat.insert_rows(0,1);

        //CHECK THAT COLUMNS OF TREETABLE ARE CORRECT
        //node_split_mat(0,0)=treetable(parent_node[0],2);
        //node_split_mat(0,1)=treetable(parent_node[0],3);

        //node_split_mat(0,0)=arma_tree_table(parent_node[0],3);
        //node_split_mat(0,1)=arma_tree_table(parent_node[0],4);

        node_split_mat(0,0)=arma_tree_table(parent_node[0],2);
        node_split_mat(0,1)=arma_tree_table(parent_node[0],3);

        node_split_mat(0,2)=rd;
        row_index=parent_node[0]+1;
        term_node=parent_node[0]+1;
      }

      //once we have the split info, loop through rows and find the subset indexes for that terminal node!
      //then fill in the predicted value for that tree
      //double prediction = tree_data(term_node,5);
      arma::uvec pred_indices;
      int split= node_split_mat(0,0)-1;

      //Rcout << "Line 224.\n";
      //Rcout << "split = " << split << ".\n";
      //arma::vec tempvec = testd.col(split);
      arma::vec tempvec = arma_orig_data.col(split);
      //Rcout << "Line 227.\n";


      double temp_split = node_split_mat(0,1);

      if(node_split_mat(0,2)==0){
        pred_indices = arma::find(tempvec <= temp_split);
      }else{
        pred_indices = arma::find(tempvec > temp_split);
      }
      //Rcout << "Line 236.\n";

      arma::uvec temp_pred_indices;

      //arma::vec data_subset = testd.col(split);
      arma::vec data_subset = arma_orig_data.col(split);

      data_subset=data_subset.elem(pred_indices);

      //now loop through each row of node_split_mat
      int n=node_split_mat.n_rows;
      //Rcout << "Line 174. i = " << i << ". n = " << n << ".\n";
      //Rcout << "Line 248.\n";

      for(int j=1;j<n;j++){
        int curr_sv=node_split_mat(j,0);
        double split_p = node_split_mat(j,1);

        //data_subset = testd.col(curr_sv-1);
        //Rcout << "Line 255.\n";
        //Rcout << "curr_sv = " << curr_sv << ".\n";
        data_subset = arma_orig_data.col(curr_sv-1);
        //Rcout << "Line 258.\n";

        data_subset=data_subset.elem(pred_indices);

        if(node_split_mat(j,2)==0){
          //split is to the left
          temp_pred_indices=arma::find(data_subset <= split_p);
        }else{
          //split is to the right
          temp_pred_indices=arma::find(data_subset > split_p);
        }
        pred_indices=pred_indices.elem(temp_pred_indices);

        if(pred_indices.size()==0){
          continue;
        }

      }
      //Rcout << "Line 199. i = " << i <<  ".\n";

      //double nodemean=tree_data(terminal_nodes[i]-1,5);
      //IntegerVector predind=as<IntegerVector>(wrap(pred_indices));
      //predictions[predind]= nodemean;
      //term_obs[i]=predind;

      double denom_temp= pred_indices.n_elem+arma::sum(alpha_pars_arma);
      //Rcout << "Line 207. predind = " << predind <<  ".\n";
      //Rcout << "Line 207. denom_temp = " << denom_temp <<  ".\n";
      // << "Line 207. term_node = " << term_node <<  ".\n";

      double num_prod=1;
      double num_sum=0;

      for(int k=0; k<num_cats; k++){
        //assuming categories of y are from 1 to num_cats
        arma::uvec cat_inds= arma::find(orig_y_arma(pred_indices)==k+1);
        double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);

        arma_tree_table(curr_term-1,5+k)= m_plus_alph/denom_temp ;

        num_prod=num_prod*tgamma(m_plus_alph);
        num_sum=num_sum +m_plus_alph ;
      }


      lik_prod= lik_prod*alph_term*num_prod/tgamma(num_sum);
      //Rcout << "Line 297.\n";


    }
    //Rcout << "Line 301.\n";

  }
  //List ret(1);
  //ret[0] = term_obs;

  //ret[0] = terminal_nodes;
  //ret[1] = term_obs;
  //ret[2] = predictions;
  //return(term_obs);
  //Rcout << "Line 309";

  //return(wrap(arma_tree_table));

  List ret(2);
  ret[0]=wrap(arma_tree_table);
  ret[1]=lik_prod;

  return(ret);

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::depends(RcppArmadillo)]]
//' @title For a set of trees, obtain tree matrices with predictions, and obtain model weights
//' @export
// [[Rcpp::export]]
List get_treelist(NumericVector original_y, int num_cats, NumericVector alpha_pars,
                  double beta_pow,
                  NumericMatrix originaldata, //NumericMatrix test_data,
                  List treetable_list//, NumericMatrix tree_data
){

  //List overall_term_nodes_trees(overall_sum_trees.size());
  //List overall_term_obs_trees(overall_sum_trees.size());
  //List overall_predictions(overall_sum_trees.size());

  List overall_treetables(treetable_list.size());
  NumericVector overall_liks(treetable_list.size());

  for(int i=0;i<treetable_list.size();i++){
    //for each set of trees loop over individual trees
    SEXP s = treetable_list[i];

    //NumericVector test_preds_sum_tree;
    if(is<List>(s)){
      //if current set of trees contains more than one tree...usually does!
      //List sum_tree=treetable_list[i];

      //save all info in list of list format the same as the trees.
      //List term_nodes_trees(sum_tree.size());
      //List term_obs_trees(sum_tree.size());
      //NumericMatrix predictions(num_obs,sum_tree.size());

      // for(int k=0;k<sum_tree.size();k++){
      //   NumericMatrix tree_table=sum_tree[k];
      //   List tree_info=get_termobs_test_data(test_data, tree_table) ;
      //   //NumericVector term_nodes=tree_info[0];
      //   //term_nodes_trees[k]=term_nodes;
      //   term_obs_trees[k]=tree_info;
      //   //umericVector term_preds=tree_info[2];
      //   //predictions(_,k)=term_preds;
      // }


      List treepred_output = get_treepreds(original_y, num_cats, alpha_pars,
                                           originaldata,
                                           treetable_list[i]  );

      overall_treetables[i]= treepred_output[0];
      double templik = as<double>(treepred_output[1]);
      overall_liks[i]= pow(templik,beta_pow);



      //overall_term_nodes_trees[i]=term_nodes_trees;
      //overall_term_obs_trees[i]= term_obs_trees;
      //overall_predictions[i]=predictions;
    }else{
      // NumericMatrix sum_tree=overall_sum_trees[i];
      // List tree_info=get_termobs_test_data(test_data, sum_tree) ;
      // //overall_term_nodes_trees[i]=tree_info[0];
      // List term_obs_trees(1);
      // term_obs_trees[0]=tree_info ;
      // //NumericVector term_preds=tree_info[2];
      // //NumericVector predictions=term_preds;
      // overall_term_obs_trees[i]= term_obs_trees;
      // //overall_predictions[i]=predictions;
      //

      //overall_treetables[i]=get_treepreds(original_y, num_cats, alpha_pars,
      //                                    originaldata,
      //                                    treetable_list[i]  );


      List treepred_output = get_treepreds(original_y, num_cats, alpha_pars,
                                           originaldata,
                                           treetable_list[i]  );

      overall_treetables[i]= treepred_output[0];
      double templik = as<double>(treepred_output[1]);
      overall_liks[i]= pow(templik,beta_pow);

    }
  }
  //List ret(1);
  //ret[0]=overall_term_nodes_trees;
  //ret[0]=overall_term_obs_trees;
  //ret[2]=overall_predictions;
  //return(overall_term_obs_trees);

  //return(overall_treetables);

  overall_liks=overall_liks/sum(overall_liks);

  List ret(2);
  ret[0]=overall_treetables;
  ret[1]=overall_liks;
  return(ret);



}
//######################################################################################################################//

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::mat get_test_probs(NumericVector weights, int num_cats,
                         NumericMatrix testdata, //NumericMatrix test_data,
                         NumericMatrix treetable//, NumericMatrix tree_data
) {
  // Function to make predictions from test data, given a single tree and the terminal node predictions, this function will be called
  //for each tree accepted in Occam's Window.

  //test_data is a nxp matrix with the same variable names as the training data the model was built on

  //tree_data is the tree table with the tree information i.e. split points and split variables and terminal node mean values

  //term_node_means is a vector storing the terminal node mean values
  // arma::vec orig_y_arma= as<arma::vec>(original_y);
  // arma::vec alpha_pars_arma= as<arma::vec>(alpha_pars);
  //
  // double lik_prod=1;
  // double alph_prod=1;
  // for(unsigned int i=0; i<alpha_pars_arma.n_elem;i++){
  //   alph_prod=alph_prod*tgamma(alpha_pars_arma(i));
  // }
  // double gam_alph_sum= tgamma(arma::sum(alpha_pars_arma));
  // double alph_term=gam_alph_sum/alph_prod;

  arma::mat arma_tree_table(treetable.begin(), treetable.nrow(), treetable.ncol(), false);
  arma::mat arma_test_data(testdata.begin(), testdata.nrow(), testdata.ncol(), false);


  //arma::mat arma_tree(tree_data.begin(), tree_data.nrow(), tree_data.ncol(), false);
  //arma::mat testd(test_data.begin(), test_data.nrow(), test_data.ncol(), false);

  //NumericVector internal_nodes=find_internal_nodes_gs(tree_data);

  NumericVector terminal_nodes=find_term_nodes(treetable);
  //arma::vec arma_terminal_nodes=Rcpp::as<arma::vec>(terminal_nodes);
  //NumericVector tree_predictions;

  //now for each internal node find the observations that belong to the terminal nodes

  //NumericVector predictions(test_data.nrow());

  arma::mat pred_mat(testdata.nrow(),num_cats);
  //arma::vec filled_in(testdata.nrow());


  //List term_obs(terminal_nodes.size());
  if(terminal_nodes.size()==1){

    //Rcout << "Line 422. \n";


    pred_mat=repmat(arma_tree_table(0,arma::span(5,5+num_cats-1)),testdata.nrow(),1);


    //Rcout << "Line 424. \n";


    // for(int k=0; k<num_cats; k++){
    // pred_mat(_,k)=rep(treetable(0,5+k),testdata.nrow());
    // }
    //double nodemean=tree_data(terminal_nodes[0]-1,5);				// let nodemean equal tree_data row terminal_nodes[i]^th row , 6th column. The minus 1 is because terminal nodes consists of indices starting at 1, but need indices to start at 0.
    //predictions=rep(nodemean,test_data.nrow());
    //Rcout << "Line 67 .\n";

    //IntegerVector temp_obsvec = seq_len(test_data.nrow())-1;
    //term_obs[0]= temp_obsvec;
    // double denom_temp= orig_y_arma.n_elem+arma::sum(alpha_pars_arma);
    //
    // double num_prod=1;
    // double num_sum=0;

    // for(int k=0; k<num_cats; k++){
    //   //assuming categories of y are from 1 to num_cats
    //   arma::uvec cat_inds= arma::find(orig_y_arma==k+1);
    //   double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
    //   arma_tree_table(0,5+k)= m_plus_alph/denom_temp ;
    //
    //   //for likelihood calculation
    //   num_prod=num_prod*tgamma(m_plus_alph);
    //   num_sum=num_sum +m_plus_alph ;
    // }
    //
    // lik_prod= alph_term*num_prod/tgamma(num_sum);
    //
  }
  else{
    for(int i=0;i<terminal_nodes.size();i++){
      //arma::mat subdata=testd;
      int curr_term=terminal_nodes[i];

      int row_index;
      int term_node=terminal_nodes[i];


      //WHAT IS THE PURPOSE OF THIS IF-STATEMENT?
      //Why should the ro index be different for a right daughter?
      //Why not just initialize row_index to any number not equal to 1 (e.g. 0)?
      row_index=0;

      // if(curr_term % 2==0){
      //   //term node is left daughter
      //   row_index=terminal_nodes[i];
      // }else{
      //   //term node is right daughter
      //   row_index=terminal_nodes[i]-1;
      // }








      //save the left and right node data into arma uvec

      //CHECK THAT THIS REFERS TO THE CORRECT COLUMNS
      //arma::vec left_nodes=arma_tree.col(0);
      //arma::vec right_nodes=arma_tree.col(1);

      arma::vec left_nodes=arma_tree_table.col(0);
      arma::vec right_nodes=arma_tree_table.col(1);



      arma::mat node_split_mat;
      node_split_mat.set_size(0,3);
      //Rcout << "Line 124. i = " << i << " .\n";

      while(row_index!=1){
        //for each terminal node work backwards and see if the parent node was a left or right node
        //append split info to a matrix
        int rd=0;
        arma::uvec parent_node=arma::find(left_nodes == term_node);

        if(parent_node.size()==0){
          parent_node=arma::find(right_nodes == term_node);
          rd=1;
        }

        //want to cout parent node and append to node_split_mat

        node_split_mat.insert_rows(0,1);

        //CHECK THAT COLUMNS OF TREETABLE ARE CORRECT
        //node_split_mat(0,0)=treetable(parent_node[0],2);
        //node_split_mat(0,1)=treetable(parent_node[0],3);

        //node_split_mat(0,0)=arma_tree_table(parent_node[0],3);
        //node_split_mat(0,1)=arma_tree_table(parent_node[0],4);

        node_split_mat(0,0)=arma_tree_table(parent_node[0],2);
        node_split_mat(0,1)=arma_tree_table(parent_node[0],3);

        node_split_mat(0,2)=rd;
        row_index=parent_node[0]+1;
        term_node=parent_node[0]+1;
      }

      //once we have the split info, loop through rows and find the subset indexes for that terminal node!
      //then fill in the predicted value for that tree
      //double prediction = tree_data(term_node,5);
      arma::uvec pred_indices;
      int split= node_split_mat(0,0)-1;

      //arma::vec tempvec = testd.col(split);
      arma::vec tempvec = arma_test_data.col(split);


      double temp_split = node_split_mat(0,1);

      if(node_split_mat(0,2)==0){
        pred_indices = arma::find(tempvec <= temp_split);
      }else{
        pred_indices = arma::find(tempvec > temp_split);
      }

      arma::uvec temp_pred_indices;

      //arma::vec data_subset = testd.col(split);
      arma::vec data_subset = arma_test_data.col(split);

      data_subset=data_subset.elem(pred_indices);

      //now loop through each row of node_split_mat
      int n=node_split_mat.n_rows;
      //Rcout << "Line 174. i = " << i << ". n = " << n << ".\n";
      //Rcout << "Line 174. node_split_mat= " << node_split_mat << ". n = " << n << ".\n";


      for(int j=1;j<n;j++){
        int curr_sv=node_split_mat(j,0);
        double split_p = node_split_mat(j,1);

        //data_subset = testd.col(curr_sv-1);
        data_subset = arma_test_data.col(curr_sv-1);

        data_subset=data_subset.elem(pred_indices);

        if(node_split_mat(j,2)==0){
          //split is to the left
          temp_pred_indices=arma::find(data_subset <= split_p);
        }else{
          //split is to the right
          temp_pred_indices=arma::find(data_subset > split_p);
        }
        pred_indices=pred_indices.elem(temp_pred_indices);

        if(pred_indices.size()==0){
          continue;
        }

      }
      //Rcout << "Line 199. i = " << i <<  ".\n";

      //double nodemean=tree_data(terminal_nodes[i]-1,5);
      //IntegerVector predind=as<IntegerVector>(wrap(pred_indices));
      //predictions[predind]= nodemean;
      //term_obs[i]=predind;

      //Rcout << "Line 635. \n";
      //Rcout << "pred_indices = " << pred_indices << ".\n";

      //pred_mat.rows(pred_indices)=arma::repmat(arma_tree_table(curr_term-1,arma::span(5,5+num_cats-1)),pred_indices.n_elem,1);
      pred_mat.each_row(pred_indices)=arma_tree_table(curr_term-1,arma::span(5,4+num_cats));



      //Rcout << "Line 588. \n";

      // for(int k=0; k<num_cats; k++){
      //   pred_mat(predind,k)=rep(treetable(curr_term-1,5+k),predind.size());
      // }


      // double denom_temp= pred_indices.n_elem+arma::sum(alpha_pars_arma);
      // //Rcout << "Line 207. predind = " << predind <<  ".\n";
      // //Rcout << "Line 207. denom_temp = " << denom_temp <<  ".\n";
      // // << "Line 207. term_node = " << term_node <<  ".\n";
      //
      // double num_prod=1;
      // double num_sum=0;
      //
      // for(int k=0; k<num_cats; k++){
      //   //assuming categories of y are from 1 to num_cats
      //   arma::uvec cat_inds= arma::find(orig_y_arma(pred_indices)==k+1);
      //   double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
      //
      //   arma_tree_table(curr_term-1,5+k)= m_plus_alph/denom_temp ;
      //
      //   num_prod=num_prod*tgamma(m_plus_alph);
      //   num_sum=num_sum +m_plus_alph ;
      // }


      //lik_prod= lik_prod*alph_term*num_prod/tgamma(num_sum);


    }
  }
  //List ret(1);
  //ret[0] = term_obs;

  //ret[0] = terminal_nodes;
  //ret[1] = term_obs;
  //ret[2] = predictions;
  //return(term_obs);

  //return(wrap(arma_tree_table));

  //List ret(2);
  //ret[0]=wrap(arma_tree_table);
  //ret[1]=lik_prod;

  return(pred_mat);

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Given tree tables and model weights, obtain predicted probabilities for test data.
//' @export
// [[Rcpp::export]]

NumericMatrix get_test_prob_overall(NumericVector weights, int num_cats,
                                    NumericMatrix testdata, //NumericMatrix test_data,
                                    List treetable_list//, NumericMatrix tree_data
){

  //List overall_term_nodes_trees(overall_sum_trees.size());
  //List overall_term_obs_trees(overall_sum_trees.size());
  //List overall_predictions(overall_sum_trees.size());

  List overall_treetables(treetable_list.size());
  NumericVector overall_liks(treetable_list.size());


  arma::mat pred_mat_overall=arma::zeros<arma::mat>(testdata.nrow(),num_cats);


  for(int i=0;i<treetable_list.size();i++){
    //for each set of trees loop over individual trees
    SEXP s = treetable_list[i];

    //NumericVector test_preds_sum_tree;
    if(is<List>(s)){
      //if current set of trees contains more than one tree...usually does!
      //List sum_tree=treetable_list[i];

      //save all info in list of list format the same as the trees.
      //List term_nodes_trees(sum_tree.size());
      //List term_obs_trees(sum_tree.size());
      //NumericMatrix predictions(num_obs,sum_tree.size());

      // for(int k=0;k<sum_tree.size();k++){
      //   NumericMatrix tree_table=sum_tree[k];
      //   List tree_info=get_termobs_test_data(test_data, tree_table) ;
      //   //NumericVector term_nodes=tree_info[0];
      //   //term_nodes_trees[k]=term_nodes;
      //   term_obs_trees[k]=tree_info;
      //   //umericVector term_preds=tree_info[2];
      //   //predictions(_,k)=term_preds;
      // }

      //Rcout << "Line 682. i== " << i << ". \n";

      arma::mat treeprob_output = get_test_probs(weights, num_cats,
                                                 testdata,
                                                 treetable_list[i]  );

      //Rcout << "Line 688. i== " << i << ". \n";

      double weighttemp = weights[i];
      //Rcout << "Line 691. i== " << i << ". \n";

      pred_mat_overall = pred_mat_overall + weighttemp*treeprob_output;
      //Rcout << "Line 694. i== " << i << ". \n";


      //overall_treetables[i]= treepred_output[0];
      //double templik = as<double>(treepred_output[1]);
      //overall_liks[i]= pow(templik,beta_pow);



      //overall_term_nodes_trees[i]=term_nodes_trees;
      //overall_term_obs_trees[i]= term_obs_trees;
      //overall_predictions[i]=predictions;
    }else{
      // NumericMatrix sum_tree=overall_sum_trees[i];
      // List tree_info=get_termobs_test_data(test_data, sum_tree) ;
      // //overall_term_nodes_trees[i]=tree_info[0];
      // List term_obs_trees(1);
      // term_obs_trees[0]=tree_info ;
      // //NumericVector term_preds=tree_info[2];
      // //NumericVector predictions=term_preds;
      // overall_term_obs_trees[i]= term_obs_trees;
      // //overall_predictions[i]=predictions;
      //

      //overall_treetables[i]=get_treepreds(original_y, num_cats, alpha_pars,
      //                                    originaldata,
      //                                    treetable_list[i]  );


      // List treepred_output = get_treepreds(original_y, num_cats, alpha_pars,
      //                                      originaldata,
      //                                      treetable_list[i]  );
      //
      // overall_treetables[i]= treepred_output[0];
      // double templik = as<double>(treepred_output[1]);
      // overall_liks[i]= pow(templik,beta_pow);


      //Rcout << "Line 732. i== " << i << ". \n";

      arma::mat treeprob_output = get_test_probs(weights, num_cats,
                                                 testdata,
                                                 treetable_list[i]  );

      //Rcout << "Line 738. i== " << i << ". \n";

      double weighttemp = weights[i];
      //Rcout << "Line 741. i== " << i << ". \n";
      //Rcout << "treeprob_output.n_rows" << treeprob_output.n_rows << ".\n";
      //Rcout << "treeprob_output.n_cols" << treeprob_output.n_cols << ".\n";


      pred_mat_overall = pred_mat_overall + weighttemp*treeprob_output;
      //Rcout << "Line 744. i== " << i << ". \n";
      //Rcout << "pred_mat_overall " << pred_mat_overall << ". \n";

    }
  }
  //List ret(1);
  //ret[0]=overall_term_nodes_trees;
  //ret[0]=overall_term_obs_trees;
  //ret[2]=overall_predictions;
  //return(overall_term_obs_trees);

  //return(overall_treetables);

  // overall_liks=overall_liks/sum(overall_liks);
  //
  // List ret(2);
  // ret[0]=overall_treetables;
  // ret[1]=overall_liks;
  // return(ret);

  return(wrap(pred_mat_overall));

}
//######################################################################################################################//

// [[Rcpp::depends(RcppArmadillo)]]
//// [[Rcpp::depends(dqrng)]]
//// [[Rcpp::depends(BH)]]
//// [[Rcpp::depends(dqrng, BH, RcppArmadillo)]]

#include <RcppArmadilloExtensions/sample.h>

//#include <dqrng.h>

//#include <boost/random/binomial_distribution.hpp>
//using binomial = boost::random::binomial_distribution<int>;
//' @title Draw a set of trees from the prior.
//' @export
// [[Rcpp::export]]
List draw_trees(double lambda, int num_trees, int seed, int num_split_vars, int num_cats ){

  //dqrng::dqRNGkind("Xoroshiro128+");
  //dqrng::dqset_seed(IntegerVector::create(seed));

  //use following with binomial?
  //dqrng::xoshiro256plus rng(seed);

  //std::vector<int> lambdavec = {lambda, 1-lambda};

  //typedef boost::mt19937 RNGType;
  //boost::random::uniform_int_distribution<> sample_splitvardist(1,num_split_vars);
  //boost::variate_generator< RNGType, boost::uniform_int<> >  sample_splitvars(rng, sample_splitvardist);

  //boost::random::uniform_real_distribution<double> b_unifdist(0,1);
  //boost::variate_generator< RNGType, boost::uniform_real<> >  b_unif_point(rng, b_unifdist);



  // std::random_device device;
  // std::mt19937 gen(device());

  //possibly use seed?
  //// std::mt19937 gen(seed);


  // std::bernoulli_distribution coin_flip(lambda);

  // std::uniform_int_distribution<> distsampvar(1, num_split_vars);
  //std::uniform_real_distribution<> dis_cont_unif(0, 1);



  List table_list(num_trees);




  for(int j=0; j<num_trees;j++){

    //If parallelizing, define the distributinos before this loop
    //and use lrng and the following two lines
    //dqrng::xoshiro256plus lrng(rng);      // make thread local copy of rng
    //lrng.jump(omp_get_thread_num() + 1);  // advance rng by 1 ... nthreads jumps


    //NumericVector treenodes_bin(0);
    //arma::uvec treenodes_bin(0);

    std::vector<int> treenodes_bin;


    int count_terminals = 0;
    int count_internals = 0;

    //int count_treebuild = 0;

    while(count_internals > (count_terminals -1)){

      //Also consider standard library and random header
      // std::random_device device;
      // std::mt19937 gen(device());
      // std::bernoulli_distribution coin_flip(lambda);
      // bool outcome = coin_flip(gen);


      //int tempdraw = coin_flip(gen);

      //int tempdraw = rbinom(n = 1, prob = lambda,size=1);


      //int tempdraw = Rcpp::rbinom(1,lambda,1);
      int tempdraw = R::rbinom(1,lambda);
      treenodes_bin.push_back(tempdraw);

      //Rcout << "tempdraw = " << tempdraw << ".\n" ;

      //int tempdraw = dqrng::dqsample_int(2, 1, true,lambdavec )-1;
      //need to update rng if use boost?
      //int tempdraw = bernoulli(rng, binomial::param_type(1, lambda));
      if(tempdraw==1){
        count_internals=count_internals+1;
      }else{
        count_terminals=count_terminals+1;
      }

    }//end of while loop creating parent vector treenodes_bin

    //Consider making this an armadillo vector
    //IntegerVector split_var_vec(treenodes_bin.size());
    //arma::uvec split_var_vec(treenodes_bin.size());
    std::vector<int> split_var_vec(treenodes_bin.size());

    //loop drawing splitting variables
    //REPLACE SQUARE BRACKETS WITH "( )" if using ARMADILLO vector for split_var_vec or treenodes_bin

    //if using armadillo, it might be faster to subset to split nodes
    //then use a vector of draws
    for(unsigned int i=0; i<treenodes_bin.size();i++){
      if(treenodes_bin[i]==0){
        split_var_vec[i] = -1;
      }else{
        // also consider the standard library function uniform_int_distribution
        // might need random header
        // This uses the Mersenne twister

        //Three lines below should probably be outside all the loops
        // std::random_device rd;
        // std::mt19937 engine(rd());
        // std::uniform_int_distribution<> distsampvar(1, num_split_vars);
        //
        // split_var_vec[i] <- distsampvar(engine);

        // split_var_vec[i] <- distsampvar(gen);


        //consider using boost
        //might need to update rng
        //split_var_vec[i] <- sample_splitvars(rng);

        //or use dqrng
        //not sure if have to update the random number
        //check if the following line is written properly
        //split_var_vec[i] = dqrng::dqsample_int(num_split_vars, 1, true);

        //not sure if this returns an integer or a vector?
        //split_var_vec[i] = RcppArmadillo::sample(num_split_vars, 1,true);
        //could try
        split_var_vec[i] = as<int>(Rcpp::sample(num_split_vars, 1,true));
        //could also try RcppArmadillo::rmultinom

      }

    }// end of for-loop drawing split variables


    //Consider making this an armadillo vector
    //NumericVector split_point_vec(treenodes_bin.size());
    //arma::vec split_point_vec(treenodes_bin.size());
    std::vector<double> split_point_vec(treenodes_bin.size());


    //loop drawing splitting points
    //REPLACE SQUARE BRACKETS WITH "( )" if using ARMADILLO vector for split_var_vec or treenodes_bin

    //if using armadillo, it might be faster to subset to split nodes
    //then use a vector of draws
    for(unsigned int i=0; i<treenodes_bin.size();i++){
      if(treenodes_bin[i]==0){
        split_point_vec[i] = -1;
      }else{


        //////////////////////////////////////////////////////////
        //following function not reccommended
        //split_point_vec[i] = std::rand();
        //////////////////////////////////////////////////////////
        ////Standard library:
        ////This should probably be outside all the loops
        ////std::random_device rd;  //Will be used to obtain a seed for the random number engine
        ////std::mt19937 gen2(rd()); //Standard mersenne_twister_engine seeded with rd()
        ////std::uniform_real_distribution<> dis_cont_unif(0, 1);

        // split_point_vec[i] = dis_cont_unif(gen);

        //////////////////////////////////////////////////////////
        //from armadillo
        split_point_vec[i] = arma::randu();

        //////////////////////////////////////////////////////////
        //probably not adviseable for paralelization
        //From Rcpp
        split_point_vec[i] = as<double>(Rcpp::runif(1,0,1));

        //////////////////////////////////////////////////////////
        //consider using boost
        //might need to update rng
        //split_point_vec[i] <- b_unif_point(rng);

        //or use dqrng
        //not sure if have to update the random number
        //check if the following line is written properly
        //split_point_vec[i] = dqrng::dqrunif(1, 0, 1);

        //not sure if this returns an integer or a vector?





      }

    }// end of for-loop drawing split points


    //Create tree table matrix

    //NumericMatrix tree_table1(treenodes_bin.size(),5+num_cats);

    //Rcout << "Line 1037. \n";
    //arma::mat tree_table1(treenodes_bin.size(),5+num_cats);

    //initialize with zeros. Not sure if this is necessary
    arma::mat tree_table1=arma::zeros<arma::mat>(treenodes_bin.size(),5+num_cats);
    //Rcout << "Line 1040. \n";


    //tree_table1(_,2) = wrap(split_var_vec);
    //tree_table1(_,3) = wrap(split_point_vec);
    //tree_table1(_,4) = wrap(treenodes_bin);

    //It might be more efficient to make everything an armadillo object initially
    // but then would need to replace push_back etc with a different approach (but this might be more efficient anyway)
    arma::colvec split_var_vec_arma=arma::conv_to<arma::colvec>::from(split_var_vec);
    arma::colvec split_point_vec_arma(split_point_vec);
    arma::colvec treenodes_bin_arma=arma::conv_to<arma::colvec>::from(treenodes_bin);


    //Rcout << "Line 1054. \n";

    tree_table1.col(2) = split_var_vec_arma;
    tree_table1.col(3) = split_point_vec_arma;
    tree_table1.col(4) = treenodes_bin_arma;


    //Rcout << "Line 1061. j = " << j << ". \n";



    // Now start filling in left daughter and right daughter columns
    std::vector<int> rd_spaces;
    int prev_node = -1;

    for(unsigned int i=0; i<treenodes_bin.size();i++){
      //Rcout << "Line 1061. i = " << i << ". \n";
      if(prev_node==0){
        //tree_table1(rd_spaces[rd_spaces.size()-1], 1)=i;
        //Rcout << "Line 1073. j = " << j << ". \n";

        tree_table1(rd_spaces.back(), 1)=i+1;
        //Rcout << "Line 1076. j = " << j << ". \n";

        rd_spaces.pop_back();
      }
      if(treenodes_bin[i]==1){
        //Rcout << "Line 1081. j = " << j << ". \n";

        tree_table1(i,0) = i+2;
        rd_spaces.push_back(i);
        prev_node = 1;
        //Rcout << "Line 185. j = " << j << ". \n";

      }else{                  // These 2 lines unnecessary if begin with matrix of zeros
        //Rcout << "Line 1089. j = " << j << ". \n";
        tree_table1(i,0)=0 ;
        tree_table1(i,1) = 0 ;
        prev_node = 0;
        //Rcout << "Line 1093. j = " << j << ". \n";

      }
    }//
    //Rcout << "Line 1097. j = " << j << ". \n";

    table_list[j]=wrap(tree_table1);
    //Rcout << "Line 1100. j = " << j << ". \n";

  }//end of loop over all trees

  return(table_list);
}//end of function definition

//######################################################################################################################//

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Safe-Bayesian Random Forest. Initial test function.
//' @export
// [[Rcpp::export]]

NumericMatrix sBayesRF(double lambda, int num_trees,
                       int seed, int num_cats,
                       NumericVector y, NumericMatrix original_datamat,
                       NumericVector alpha_parameters, double beta_par,
                       NumericMatrix test_datamat){

  int num_split_vars= original_datamat.ncol();
  NumericMatrix Data_transformed = cpptrans_cdf(original_datamat);
  NumericMatrix testdat_trans = cpptrans_cdf_test(original_datamat,test_datamat);

  //Rcout << "Line 1134 . \n";
  List table_list = draw_trees(lambda, num_trees, seed, num_split_vars, num_cats );
  //Rcout << "Line 1136 . \n";


  List tree_list_output = get_treelist(y, num_cats, alpha_parameters, beta_par,
                                       Data_transformed,
                                       table_list  );
  //Rcout << "Line 1141 . \n";

  NumericMatrix probmat = get_test_prob_overall(tree_list_output[1],num_cats,
                                                testdat_trans,
                                                tree_list_output[0]);
  //Rcout << "Line 1146 . \n";

  return(probmat);

}
//######################################################################################################################//

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Safe-Bayesian Random Forest
//'
//' @description An implementation of the Safe-Bayesian Random Forest described by Quadrianto and Ghahramani (2015)
//' @param lambda A real number between 0 and 1 that determines the splitting probability in the prior (which is used as the importance sampler of tree models). Quadrianto and Ghahramani (2015) recommend a value less than 0.5 .
//' @param num_trees The number of trees to be sampled.
//' @param seed The seed for random number generation.
//' @param num_cats The number of possible values for the outcome variable.
//' @param y The training data vector of outcomes. This must be a vector of integers between 1 and num_cats.
//' @param original_datamat The original training data. Currently all variables must be continuous. The training data does not need to be transformed before being entered to this function.
//' @param alpha_parameters Vector of prior parameters.
//' @param beta_par The power to which the likelihood is to be raised. For BMA, set beta_par=1.
//' @param original_datamat The original test data. This matrix must have the same number of columns (variables) as the training data. Currently all variables must be continuous. The test data does not need to be transformed before being entered to this function.
//' @export
// [[Rcpp::export]]

NumericMatrix sBayesRF_onefunc(double lambda, int num_trees,
                               int seed, int num_cats,
                               NumericVector y, NumericMatrix original_datamat,
                               NumericVector alpha_parameters, double beta_par,
                               NumericMatrix test_datamat){

  int num_split_vars= original_datamat.ncol();


  ///////////////////////
  //NumericMatrix Data_transformed = cpptrans_cdf(original_datamat);
  NumericMatrix Data_transformed(original_datamat.nrow(), original_datamat.ncol());
  for(int i=0; i<original_datamat.ncol();i++){
    NumericVector samp= original_datamat(_,i);
    NumericVector sv(clone(samp));
    std::sort(sv.begin(), sv.end());
    double nobs = samp.size();
    NumericVector ans(nobs);
    for (int k = 0; k < samp.size(); ++k)
      ans[k] = std::lower_bound(sv.begin(), sv.end(), samp[k]) - sv.begin();
    //NumericVector ansnum = ans;
    Data_transformed(_,i) = (ans+1)/nobs;
  }



  /////////////////////////////////////
  //NumericMatrix testdat_trans = cpptrans_cdf_test(original_datamat,test_datamat);
  NumericMatrix testdat_trans(test_datamat.nrow(), test_datamat.ncol());
  for(int i=0; i<test_datamat.ncol();i++){
    NumericVector samp= test_datamat(_,i);
    NumericVector svtest = original_datamat(_,i);
    NumericVector sv(clone(svtest));
    std::sort(sv.begin(), sv.end());
    double nobs = samp.size();
    NumericVector ans(nobs);
    double nobsref = svtest.size();
    for (int k = 0; k < samp.size(); ++k){
      ans[k] = std::lower_bound(sv.begin(), sv.end(), samp[k]) - sv.begin();
    }
    //NumericVector ansnum = ans;
    testdat_trans(_,i) = (ans)/nobsref;
  }



  /////////////////////////////////////////////////////////////////////////////////////////



  //////////////////////////////////////////////////////////////////////////////////////
  //List table_list = draw_trees(lambda, num_trees, seed, num_split_vars, num_cats );



  //dqrng::dqRNGkind("Xoroshiro128+");
  //dqrng::dqset_seed(IntegerVector::create(seed));

  //use following with binomial?
  //dqrng::xoshiro256plus rng(seed);

  //std::vector<int> lambdavec = {lambda, 1-lambda};

  //typedef boost::mt19937 RNGType;
  //boost::random::uniform_int_distribution<> sample_splitvardist(1,num_split_vars);
  //boost::variate_generator< RNGType, boost::uniform_int<> >  sample_splitvars(rng, sample_splitvardist);

  //boost::random::uniform_real_distribution<double> b_unifdist(0,1);
  //boost::variate_generator< RNGType, boost::uniform_real<> >  b_unif_point(rng, b_unifdist);



  // std::random_device device;
  // std::mt19937 gen(device());

  //possibly use seed?
  //// std::mt19937 gen(seed);


  // std::bernoulli_distribution coin_flip(lambda);

  // std::uniform_int_distribution<> distsampvar(1, num_split_vars);
  //std::uniform_real_distribution<> dis_cont_unif(0, 1);


  arma::vec orig_y_arma= as<arma::vec>(y);
  arma::vec alpha_pars_arma= as<arma::vec>(alpha_parameters);

  arma::mat arma_orig_data(Data_transformed.begin(), Data_transformed.nrow(), Data_transformed.ncol(), false);
  arma::mat arma_test_data(testdat_trans.begin(), testdat_trans.nrow(), testdat_trans.ncol(), false);


  arma::mat pred_mat_overall=arma::zeros<arma::mat>(test_datamat.nrow(),num_cats);


  //List overall_treetables(num_trees);
  NumericVector overall_liks(num_trees);


  //overall_treetables[i]= wrap(tree_table1);
  //double templik = as<double>(treepred_output[1]);
  //overall_liks[i]= pow(lik_prod,beta_pow);



  for(int j=0; j<num_trees;j++){

    //If parallelizing, define the distributinos before this loop
    //and use lrng and the following two lines
    //dqrng::xoshiro256plus lrng(rng);      // make thread local copy of rng
    //lrng.jump(omp_get_thread_num() + 1);  // advance rng by 1 ... nthreads jumps


    //NumericVector treenodes_bin(0);
    //arma::uvec treenodes_bin(0);

    std::vector<int> treenodes_bin;


    int count_terminals = 0;
    int count_internals = 0;

    //int count_treebuild = 0;

    while(count_internals > (count_terminals -1)){

      //Also consider standard library and random header
      // std::random_device device;
      // std::mt19937 gen(device());
      // std::bernoulli_distribution coin_flip(lambda);
      // bool outcome = coin_flip(gen);


      //int tempdraw = coin_flip(gen);

      //int tempdraw = rbinom(n = 1, prob = lambda,size=1);


      //int tempdraw = Rcpp::rbinom(1,lambda,1);
      int tempdraw = R::rbinom(1,lambda);
      treenodes_bin.push_back(tempdraw);

      //Rcout << "tempdraw = " << tempdraw << ".\n" ;

      //int tempdraw = dqrng::dqsample_int(2, 1, true,lambdavec )-1;
      //need to update rng if use boost?
      //int tempdraw = bernoulli(rng, binomial::param_type(1, lambda));
      if(tempdraw==1){
        count_internals=count_internals+1;
      }else{
        count_terminals=count_terminals+1;
      }

    }//end of while loop creating parent vector treenodes_bin

    //Consider making this an armadillo vector
    //IntegerVector split_var_vec(treenodes_bin.size());
    //arma::uvec split_var_vec(treenodes_bin.size());
    std::vector<int> split_var_vec(treenodes_bin.size());

    //loop drawing splitting variables
    //REPLACE SQUARE BRACKETS WITH "( )" if using ARMADILLO vector for split_var_vec or treenodes_bin

    //if using armadillo, it might be faster to subset to split nodes
    //then use a vector of draws
    for(unsigned int i=0; i<treenodes_bin.size();i++){
      if(treenodes_bin[i]==0){
        split_var_vec[i] = -1;
      }else{
        // also consider the standard library function uniform_int_distribution
        // might need random header
        // This uses the Mersenne twister

        //Three lines below should probably be outside all the loops
        // std::random_device rd;
        // std::mt19937 engine(rd());
        // std::uniform_int_distribution<> distsampvar(1, num_split_vars);
        //
        // split_var_vec[i] <- distsampvar(engine);

        // split_var_vec[i] <- distsampvar(gen);


        //consider using boost
        //might need to update rng
        //split_var_vec[i] <- sample_splitvars(rng);

        //or use dqrng
        //not sure if have to update the random number
        //check if the following line is written properly
        //split_var_vec[i] = dqrng::dqsample_int(num_split_vars, 1, true);

        //not sure if this returns an integer or a vector?
        //split_var_vec[i] = RcppArmadillo::sample(num_split_vars, 1,true);
        //could try
        split_var_vec[i] = as<int>(Rcpp::sample(num_split_vars, 1,true));
        //could also try RcppArmadillo::rmultinom

      }

    }// end of for-loop drawing split variables


    //Consider making this an armadillo vector
    //NumericVector split_point_vec(treenodes_bin.size());
    //arma::vec split_point_vec(treenodes_bin.size());
    std::vector<double> split_point_vec(treenodes_bin.size());


    //loop drawing splitting points
    //REPLACE SQUARE BRACKETS WITH "( )" if using ARMADILLO vector for split_var_vec or treenodes_bin

    //if using armadillo, it might be faster to subset to split nodes
    //then use a vector of draws
    for(unsigned int i=0; i<treenodes_bin.size();i++){
      if(treenodes_bin[i]==0){
        split_point_vec[i] = -1;
      }else{


        //////////////////////////////////////////////////////////
        //following function not reccommended
        //split_point_vec[i] = std::rand();
        //////////////////////////////////////////////////////////
        ////Standard library:
        ////This should probably be outside all the loops
        ////std::random_device rd;  //Will be used to obtain a seed for the random number engine
        ////std::mt19937 gen2(rd()); //Standard mersenne_twister_engine seeded with rd()
        ////std::uniform_real_distribution<> dis_cont_unif(0, 1);

        // split_point_vec[i] = dis_cont_unif(gen);

        //////////////////////////////////////////////////////////
        //from armadillo
        //split_point_vec[i] = arma::randu();

        //////////////////////////////////////////////////////////
        //probably not adviseable for paralelization
        //From Rcpp
        split_point_vec[i] = as<double>(Rcpp::runif(1,0,1));

        //////////////////////////////////////////////////////////
        //consider using boost
        //might need to update rng
        //split_point_vec[i] <- b_unif_point(rng);

        //or use dqrng
        //not sure if have to update the random number
        //check if the following line is written properly
        //split_point_vec[i] = dqrng::dqrunif(1, 0, 1);

        //not sure if this returns an integer or a vector?





      }

    }// end of for-loop drawing split points


    //Create tree table matrix

    //NumericMatrix tree_table1(treenodes_bin.size(),5+num_cats);

    //Rcout << "Line 1037. \n";
    //arma::mat tree_table1(treenodes_bin.size(),5+num_cats);

    //initialize with zeros. Not sure if this is necessary
    arma::mat tree_table1=arma::zeros<arma::mat>(treenodes_bin.size(),5+num_cats);
    //Rcout << "Line 1040. \n";


    //tree_table1(_,2) = wrap(split_var_vec);
    //tree_table1(_,3) = wrap(split_point_vec);
    //tree_table1(_,4) = wrap(treenodes_bin);

    //It might be more efficient to make everything an armadillo object initially
    // but then would need to replace push_back etc with a different approach (but this might be more efficient anyway)
    arma::colvec split_var_vec_arma=arma::conv_to<arma::colvec>::from(split_var_vec);
    arma::colvec split_point_vec_arma(split_point_vec);
    arma::colvec treenodes_bin_arma=arma::conv_to<arma::colvec>::from(treenodes_bin);


    //Rcout << "Line 1054. \n";

    tree_table1.col(2) = split_var_vec_arma;
    tree_table1.col(3) = split_point_vec_arma;
    tree_table1.col(4) = treenodes_bin_arma;


    //Rcout << "Line 1061. j = " << j << ". \n";



    // Now start filling in left daughter and right daughter columns
    std::vector<int> rd_spaces;
    int prev_node = -1;

    for(unsigned int i=0; i<treenodes_bin.size();i++){
      //Rcout << "Line 1061. i = " << i << ". \n";
      if(prev_node==0){
        //tree_table1(rd_spaces[rd_spaces.size()-1], 1)=i;
        //Rcout << "Line 1073. j = " << j << ". \n";

        tree_table1(rd_spaces.back(), 1)=i+1;
        //Rcout << "Line 1076. j = " << j << ". \n";

        rd_spaces.pop_back();
      }
      if(treenodes_bin[i]==1){
        //Rcout << "Line 1081. j = " << j << ". \n";

        tree_table1(i,0) = i+2;
        rd_spaces.push_back(i);
        prev_node = 1;
        //Rcout << "Line 185. j = " << j << ". \n";

      }else{                  // These 2 lines unnecessary if begin with matrix of zeros
        //Rcout << "Line 1089. j = " << j << ". \n";
        tree_table1(i,0)=0 ;
        tree_table1(i,1) = 0 ;
        prev_node = 0;
        //Rcout << "Line 1093. j = " << j << ". \n";

      }
    }//
    //Rcout << "Line 1097. j = " << j << ". \n";





    //List treepred_output = get_treepreds(original_y, num_cats, alpha_pars,
    //                                     originaldata,
    //                                     treetable_list[i]  );


    //use armadillo object tree_table1

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////



    double lik_prod=1;
    double alph_prod=1;
    for(unsigned int i=0; i<alpha_pars_arma.n_elem;i++){
      alph_prod=alph_prod*tgamma(alpha_pars_arma(i));
    }
    double gam_alph_sum= tgamma(arma::sum(alpha_pars_arma));
    double alph_term=gam_alph_sum/alph_prod;

    //arma::mat arma_tree_table(treetable.begin(), treetable.nrow(), treetable.ncol(), false);
    //arma::mat arma_orig_data(originaldata.begin(), originaldata.nrow(), originaldata.ncol(), false);


    //arma::mat arma_tree(tree_data.begin(), tree_data.nrow(), tree_data.ncol(), false);
    //arma::mat testd(test_data.begin(), test_data.nrow(), test_data.ncol(), false);

    //NumericVector internal_nodes=find_internal_nodes_gs(tree_data);

    //NumericVector terminal_nodes=find_term_nodes(treetable);

    //arma::mat arma_tree(tree_table.begin(),tree_table.nrow(), tree_table.ncol(), false);

    //arma::vec colmat=arma_tree.col(4);
    //arma::uvec term_nodes=arma::find(colmat==-1);

    //arma::vec colmat=arma_tree.col(2);
    //arma::uvec term_nodes=arma::find(colmat==0);

    arma::vec colmat=tree_table1.col(4);
    arma::uvec term_nodes=arma::find(colmat==0);

    term_nodes=term_nodes+1;

    NumericVector terminal_nodes= wrap(term_nodes);




    //arma::vec arma_terminal_nodes=Rcpp::as<arma::vec>(terminal_nodes);
    //NumericVector tree_predictions;

    //now for each internal node find the observations that belong to the terminal nodes

    //NumericVector predictions(test_data.nrow());
    //List term_obs(terminal_nodes.size());
    if(terminal_nodes.size()==1){
      //double nodemean=tree_data(terminal_nodes[0]-1,5);				// let nodemean equal tree_data row terminal_nodes[i]^th row , 6th column. The minus 1 is because terminal nodes consists of indices starting at 1, but need indices to start at 0.
      //predictions=rep(nodemean,test_data.nrow());
      //Rcout << "Line 67 .\n";

      //IntegerVector temp_obsvec = seq_len(test_data.nrow())-1;
      //term_obs[0]= temp_obsvec;
      double denom_temp= orig_y_arma.n_elem+arma::sum(alpha_pars_arma);

      double num_prod=1;
      double num_sum=0;
      //Rcout << "Line 129.\n";

      for(int k=0; k<num_cats; k++){
        //assuming categories of y are from 1 to num_cats
        arma::uvec cat_inds= arma::find(orig_y_arma==k+1);
        double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
        tree_table1(0,5+k)= m_plus_alph/denom_temp ;

        //for likelihood calculation
        num_prod=num_prod*tgamma(m_plus_alph);
        num_sum=num_sum +m_plus_alph ;
      }

      lik_prod= alph_term*num_prod/tgamma(num_sum);

    }
    else{
      for(int i=0;i<terminal_nodes.size();i++){
        //arma::mat subdata=testd;
        int curr_term=terminal_nodes[i];

        int row_index;
        int term_node=terminal_nodes[i];
        //Rcout << "Line 152.\n";


        //WHAT IS THE PURPOSE OF THIS IF-STATEMENT?
        //Why should the ro index be different for a right daughter?
        //Why not just initialize row_index to any number not equal to 1 (e.g. 0)?
        row_index=0;

        // if(curr_term % 2==0){
        //   //term node is left daughter
        //   row_index=terminal_nodes[i];
        // }else{
        //   //term node is right daughter
        //   row_index=terminal_nodes[i]-1;
        // }




        //save the left and right node data into arma uvec

        //CHECK THAT THIS REFERS TO THE CORRECT COLUMNS
        //arma::vec left_nodes=arma_tree.col(0);
        //arma::vec right_nodes=arma_tree.col(1);

        arma::vec left_nodes=tree_table1.col(0);
        arma::vec right_nodes=tree_table1.col(1);



        arma::mat node_split_mat;
        node_split_mat.set_size(0,3);
        //Rcout << "Line 182. i = " << i << " .\n";

        while(row_index!=1){
          //for each terminal node work backwards and see if the parent node was a left or right node
          //append split info to a matrix
          int rd=0;
          arma::uvec parent_node=arma::find(left_nodes == term_node);

          if(parent_node.size()==0){
            parent_node=arma::find(right_nodes == term_node);
            rd=1;
          }

          //want to cout parent node and append to node_split_mat

          node_split_mat.insert_rows(0,1);

          //CHECK THAT COLUMNS OF TREETABLE ARE CORRECT
          //node_split_mat(0,0)=treetable(parent_node[0],2);
          //node_split_mat(0,1)=treetable(parent_node[0],3);

          //node_split_mat(0,0)=arma_tree_table(parent_node[0],3);
          //node_split_mat(0,1)=arma_tree_table(parent_node[0],4);

          node_split_mat(0,0)=tree_table1(parent_node[0],2);
          node_split_mat(0,1)=tree_table1(parent_node[0],3);

          node_split_mat(0,2)=rd;
          row_index=parent_node[0]+1;
          term_node=parent_node[0]+1;
        }

        //once we have the split info, loop through rows and find the subset indexes for that terminal node!
        //then fill in the predicted value for that tree
        //double prediction = tree_data(term_node,5);
        arma::uvec pred_indices;
        int split= node_split_mat(0,0)-1;

        //Rcout << "Line 224.\n";
        //Rcout << "split = " << split << ".\n";
        //arma::vec tempvec = testd.col(split);
        arma::vec tempvec = arma_orig_data.col(split);
        //Rcout << "Line 227.\n";


        double temp_split = node_split_mat(0,1);

        if(node_split_mat(0,2)==0){
          pred_indices = arma::find(tempvec <= temp_split);
        }else{
          pred_indices = arma::find(tempvec > temp_split);
        }
        //Rcout << "Line 236.\n";

        arma::uvec temp_pred_indices;

        //arma::vec data_subset = testd.col(split);
        arma::vec data_subset = arma_orig_data.col(split);

        data_subset=data_subset.elem(pred_indices);

        //now loop through each row of node_split_mat
        int n=node_split_mat.n_rows;
        //Rcout << "Line 174. i = " << i << ". n = " << n << ".\n";
        //Rcout << "Line 248.\n";

        for(int j=1;j<n;j++){
          int curr_sv=node_split_mat(j,0);
          double split_p = node_split_mat(j,1);

          //data_subset = testd.col(curr_sv-1);
          //Rcout << "Line 255.\n";
          //Rcout << "curr_sv = " << curr_sv << ".\n";
          data_subset = arma_orig_data.col(curr_sv-1);
          //Rcout << "Line 258.\n";

          data_subset=data_subset.elem(pred_indices);

          if(node_split_mat(j,2)==0){
            //split is to the left
            temp_pred_indices=arma::find(data_subset <= split_p);
          }else{
            //split is to the right
            temp_pred_indices=arma::find(data_subset > split_p);
          }
          pred_indices=pred_indices.elem(temp_pred_indices);

          if(pred_indices.size()==0){
            continue;
          }

        }
        //Rcout << "Line 199. i = " << i <<  ".\n";

        //double nodemean=tree_data(terminal_nodes[i]-1,5);
        //IntegerVector predind=as<IntegerVector>(wrap(pred_indices));
        //predictions[predind]= nodemean;
        //term_obs[i]=predind;

        double denom_temp= pred_indices.n_elem+arma::sum(alpha_pars_arma);
        //Rcout << "Line 207. predind = " << predind <<  ".\n";
        //Rcout << "Line 207. denom_temp = " << denom_temp <<  ".\n";
        // << "Line 207. term_node = " << term_node <<  ".\n";

        double num_prod=1;
        double num_sum=0;

        for(int k=0; k<num_cats; k++){
          //assuming categories of y are from 1 to num_cats
          arma::uvec cat_inds= arma::find(orig_y_arma(pred_indices)==k+1);
          double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);

          tree_table1(curr_term-1,5+k)= m_plus_alph/denom_temp ;

          num_prod=num_prod*tgamma(m_plus_alph);
          num_sum=num_sum +m_plus_alph ;
        }


        lik_prod= lik_prod*alph_term*num_prod/tgamma(num_sum);
        //Rcout << "Line 297.\n";


      }
      //Rcout << "Line 301.\n";

    }
    //List ret(1);
    //ret[0] = term_obs;

    //ret[0] = terminal_nodes;
    //ret[1] = term_obs;
    //ret[2] = predictions;
    //return(term_obs);
    //Rcout << "Line 309";

    //return(wrap(arma_tree_table));

    //List ret(2);
    //ret[0]=wrap(arma_tree_table);
    //ret[1]=lik_prod;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////






    //overall_treetables[j]= wrap(tree_table1);


    //double templik = as<double>(treepred_output[1]);

    double templik = pow(lik_prod,beta_par);
    overall_liks[j]= templik;






    //arma::mat arma_tree_table(treetable.begin(), treetable.nrow(), treetable.ncol(), false);
    //arma::mat arma_test_data(testdata.begin(), testdata.nrow(), testdata.ncol(), false);


    //arma::mat arma_tree(tree_data.begin(), tree_data.nrow(), tree_data.ncol(), false);
    //arma::mat testd(test_data.begin(), test_data.nrow(), test_data.ncol(), false);

    //NumericVector internal_nodes=find_internal_nodes_gs(tree_data);

    //NumericVector terminal_nodes=find_term_nodes(treetable);
    //arma::vec arma_terminal_nodes=Rcpp::as<arma::vec>(terminal_nodes);
    //NumericVector tree_predictions;

    //now for each internal node find the observations that belong to the terminal nodes

    //NumericVector predictions(test_data.nrow());

    arma::mat pred_mat(test_datamat.nrow(),num_cats);
    //arma::vec filled_in(testdata.nrow());


    //List term_obs(terminal_nodes.size());
    if(terminal_nodes.size()==1){

      //Rcout << "Line 422. \n";


      pred_mat=repmat(tree_table1(0,arma::span(5,5+num_cats-1)),test_datamat.nrow(),1);


      //Rcout << "Line 424. \n";


      // for(int k=0; k<num_cats; k++){
      // pred_mat(_,k)=rep(treetable(0,5+k),testdata.nrow());
      // }
      //double nodemean=tree_data(terminal_nodes[0]-1,5);				// let nodemean equal tree_data row terminal_nodes[i]^th row , 6th column. The minus 1 is because terminal nodes consists of indices starting at 1, but need indices to start at 0.
      //predictions=rep(nodemean,test_data.nrow());
      //Rcout << "Line 67 .\n";

      //IntegerVector temp_obsvec = seq_len(test_data.nrow())-1;
      //term_obs[0]= temp_obsvec;
      // double denom_temp= orig_y_arma.n_elem+arma::sum(alpha_pars_arma);
      //
      // double num_prod=1;
      // double num_sum=0;

      // for(int k=0; k<num_cats; k++){
      //   //assuming categories of y are from 1 to num_cats
      //   arma::uvec cat_inds= arma::find(orig_y_arma==k+1);
      //   double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
      //   arma_tree_table(0,5+k)= m_plus_alph/denom_temp ;
      //
      //   //for likelihood calculation
      //   num_prod=num_prod*tgamma(m_plus_alph);
      //   num_sum=num_sum +m_plus_alph ;
      // }
      //
      // lik_prod= alph_term*num_prod/tgamma(num_sum);
      //
    }
    else{
      for(int i=0;i<terminal_nodes.size();i++){
        //arma::mat subdata=testd;
        int curr_term=terminal_nodes[i];

        int row_index;
        int term_node=terminal_nodes[i];


        //WHAT IS THE PURPOSE OF THIS IF-STATEMENT?
        //Why should the ro index be different for a right daughter?
        //Why not just initialize row_index to any number not equal to 1 (e.g. 0)?
        row_index=0;

        // if(curr_term % 2==0){
        //   //term node is left daughter
        //   row_index=terminal_nodes[i];
        // }else{
        //   //term node is right daughter
        //   row_index=terminal_nodes[i]-1;
        // }








        //save the left and right node data into arma uvec

        //CHECK THAT THIS REFERS TO THE CORRECT COLUMNS
        //arma::vec left_nodes=arma_tree.col(0);
        //arma::vec right_nodes=arma_tree.col(1);

        arma::vec left_nodes=tree_table1.col(0);
        arma::vec right_nodes=tree_table1.col(1);



        arma::mat node_split_mat;
        node_split_mat.set_size(0,3);
        //Rcout << "Line 124. i = " << i << " .\n";

        while(row_index!=1){
          //for each terminal node work backwards and see if the parent node was a left or right node
          //append split info to a matrix
          int rd=0;
          arma::uvec parent_node=arma::find(left_nodes == term_node);

          if(parent_node.size()==0){
            parent_node=arma::find(right_nodes == term_node);
            rd=1;
          }

          //want to cout parent node and append to node_split_mat

          node_split_mat.insert_rows(0,1);

          //CHECK THAT COLUMNS OF TREETABLE ARE CORRECT
          //node_split_mat(0,0)=treetable(parent_node[0],2);
          //node_split_mat(0,1)=treetable(parent_node[0],3);

          //node_split_mat(0,0)=arma_tree_table(parent_node[0],3);
          //node_split_mat(0,1)=arma_tree_table(parent_node[0],4);

          node_split_mat(0,0)=tree_table1(parent_node[0],2);
          node_split_mat(0,1)=tree_table1(parent_node[0],3);

          node_split_mat(0,2)=rd;
          row_index=parent_node[0]+1;
          term_node=parent_node[0]+1;
        }

        //once we have the split info, loop through rows and find the subset indexes for that terminal node!
        //then fill in the predicted value for that tree
        //double prediction = tree_data(term_node,5);
        arma::uvec pred_indices;
        int split= node_split_mat(0,0)-1;

        //arma::vec tempvec = testd.col(split);
        arma::vec tempvec = arma_test_data.col(split);


        double temp_split = node_split_mat(0,1);

        if(node_split_mat(0,2)==0){
          pred_indices = arma::find(tempvec <= temp_split);
        }else{
          pred_indices = arma::find(tempvec > temp_split);
        }

        arma::uvec temp_pred_indices;

        //arma::vec data_subset = testd.col(split);
        arma::vec data_subset = arma_test_data.col(split);

        data_subset=data_subset.elem(pred_indices);

        //now loop through each row of node_split_mat
        int n=node_split_mat.n_rows;
        //Rcout << "Line 174. i = " << i << ". n = " << n << ".\n";
        //Rcout << "Line 174. node_split_mat= " << node_split_mat << ". n = " << n << ".\n";


        for(int j=1;j<n;j++){
          int curr_sv=node_split_mat(j,0);
          double split_p = node_split_mat(j,1);

          //data_subset = testd.col(curr_sv-1);
          data_subset = arma_test_data.col(curr_sv-1);

          data_subset=data_subset.elem(pred_indices);

          if(node_split_mat(j,2)==0){
            //split is to the left
            temp_pred_indices=arma::find(data_subset <= split_p);
          }else{
            //split is to the right
            temp_pred_indices=arma::find(data_subset > split_p);
          }
          pred_indices=pred_indices.elem(temp_pred_indices);

          if(pred_indices.size()==0){
            continue;
          }

        }
        //Rcout << "Line 199. i = " << i <<  ".\n";

        //double nodemean=tree_data(terminal_nodes[i]-1,5);
        //IntegerVector predind=as<IntegerVector>(wrap(pred_indices));
        //predictions[predind]= nodemean;
        //term_obs[i]=predind;

        //Rcout << "Line 635. \n";
        //Rcout << "pred_indices = " << pred_indices << ".\n";

        //pred_mat.rows(pred_indices)=arma::repmat(arma_tree_table(curr_term-1,arma::span(5,5+num_cats-1)),pred_indices.n_elem,1);
        pred_mat.each_row(pred_indices)=tree_table1(curr_term-1,arma::span(5,4+num_cats));



        //Rcout << "Line 588. \n";

        // for(int k=0; k<num_cats; k++){
        //   pred_mat(predind,k)=rep(treetable(curr_term-1,5+k),predind.size());
        // }


        // double denom_temp= pred_indices.n_elem+arma::sum(alpha_pars_arma);
        // //Rcout << "Line 207. predind = " << predind <<  ".\n";
        // //Rcout << "Line 207. denom_temp = " << denom_temp <<  ".\n";
        // // << "Line 207. term_node = " << term_node <<  ".\n";
        //
        // double num_prod=1;
        // double num_sum=0;
        //
        // for(int k=0; k<num_cats; k++){
        //   //assuming categories of y are from 1 to num_cats
        //   arma::uvec cat_inds= arma::find(orig_y_arma(pred_indices)==k+1);
        //   double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
        //
        //   arma_tree_table(curr_term-1,5+k)= m_plus_alph/denom_temp ;
        //
        //   num_prod=num_prod*tgamma(m_plus_alph);
        //   num_sum=num_sum +m_plus_alph ;
        // }


        //lik_prod= lik_prod*alph_term*num_prod/tgamma(num_sum);


      }
    }






    //THIS SHOULD BE DIFFERENT IF THE CODE IS TO BE PARALLELIZED
    //EACH THREAD SHOULD OUTPUT ITS OWN MATRIX AND SUM OF LIKELIHOODS
    //THEN ADD THE MATRICES TOGETHER AND DIVIDE BY THE TOTAL SUM OF LIKELIHOODS
    //OR JUST SAVE ALL MATRICES TO ONE LIST

    pred_mat_overall = pred_mat_overall + templik*pred_mat;




    //arma::mat treeprob_output = get_test_probs(weights, num_cats,
    //                                           testdata,
    //                                           treetable_list[i]  );

    //Rcout << "Line 688. i== " << i << ". \n";

    //double weighttemp = weights[i];
    //Rcout << "Line 691. i== " << i << ". \n";

    //pred_mat_overall = pred_mat_overall + weighttemp*treeprob_output;



  }//end of loop over all trees




  ///////////////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////////



  double sumlik_total= sum(overall_liks);
  pred_mat_overall=pred_mat_overall*(1/sumlik_total);
  //Rcout << "Line 1141 . \n";
  //Rcout << "Line 1146 . \n";

  return(wrap(pred_mat_overall));

}
//######################################################################################################################//

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Safe-Bayesian Random Forest in C++
//'
//' @description An implementation of the Safe-Bayesian Random Forest described by Quadrianto and Ghahramani (2015)
//' @param lambda A real number between 0 and 1 that determines the splitting probability in the prior (which is used as the importance sampler of tree models). Quadrianto and Ghahramani (2015) recommend a value less than 0.5 .
//' @param num_trees The number of trees to be sampled.
//' @param seed The seed for random number generation.
//' @param num_cats The number of possible values for the outcome variable.
//' @param y The training data vector of outcomes. This must be a vector of integers between 1 and num_cats.
//' @param original_datamat The original training data. Currently all variables must be continuous. The training data does not need to be transformed before being entered to this function.
//' @param alpha_parameters Vector of prior parameters.
//' @param beta_par The power to which the likelihood is to be raised. For BMA, set beta_par=1.
//' @param original_datamat The original test data. This matrix must have the same number of columns (variables) as the training data. Currently all variables must be continuous. The test data does not need to be transformed before being entered to this function.
//' @export
// [[Rcpp::export]]

NumericMatrix sBayesRF_onefunc_arma(double lambda, int num_trees,
                                    int seed, int num_cats,
                                    NumericVector y, NumericMatrix original_datamat,
                                    NumericVector alpha_parameters, double beta_par,
                                    NumericMatrix test_datamat){

  int num_split_vars= original_datamat.ncol();
  arma::mat data_arma= as<arma::mat>(original_datamat);
  arma::mat testdata_arma= as<arma::mat>(test_datamat);
  arma::vec orig_y_arma= as<arma::vec>(y);
  arma::vec alpha_pars_arma= as<arma::vec>(alpha_parameters);

  ///////////////////////
  //NumericMatrix Data_transformed = cpptrans_cdf(original_datamat);
  // NumericMatrix Data_transformed(original_datamat.nrow(), original_datamat.ncol());
  // for(int i=0; i<original_datamat.ncol();i++){
  //   NumericVector samp= original_datamat(_,i);
  //   NumericVector sv(clone(samp));
  //   std::sort(sv.begin(), sv.end());
  //   double nobs = samp.size();
  //   NumericVector ans(nobs);
  //   for (int k = 0; k < samp.size(); ++k)
  //     ans[k] = std::lower_bound(sv.begin(), sv.end(), samp[k]) - sv.begin();
  //   //NumericVector ansnum = ans;
  //   Data_transformed(_,i) = (ans+1)/nobs;
  // }



  //arma::mat arma_orig_data(Data_transformed.begin(), Data_transformed.nrow(), Data_transformed.ncol(), false);



  //NumericMatrix transformedData(originaldata.nrow(), originaldata.ncol());

  //THIS CAN BE PARALLELIZED IF THERE ARE MANY VARIABLES
  arma::mat arma_orig_data(data_arma.n_rows,data_arma.n_cols);
  for(unsigned int k=0; k<data_arma.n_cols;k++){
    arma::vec samp= data_arma.col(k);
    arma::vec sv=arma::sort(samp);
    //std::sort(sv.begin(), sv.end());
    arma::uvec ord = arma::sort_index(samp);
    double nobs = samp.n_elem;
    arma::vec ans(nobs);
    for (unsigned int i = 0, j = 0; i < nobs; ++i) {
      int ind=ord(i);
      double ssampi(samp[ind]);
      while (sv(j) < ssampi && j < sv.size()) ++j;
      ans(ind) = j;     // j is the 1-based index of the lower bound
    }
    arma_orig_data.col(k)=(ans+1)/nobs;
  }





  /////////////////////////////////////
  // NumericMatrix testdat_trans = cpptrans_cdf_test(original_datamat,test_datamat);
  // //NumericMatrix testdat_trans(test_datamat.nrow(), test_datamat.ncol());
  // for(int i=0; i<test_datamat.ncol();i++){
  //   NumericVector samp= test_datamat(_,i);
  //   NumericVector svtest = original_datamat(_,i);
  //   NumericVector sv(clone(svtest));
  //   std::sort(sv.begin(), sv.end());
  //   double nobs = samp.size();
  //   NumericVector ans(nobs);
  //   double nobsref = svtest.size();
  //   for (int k = 0; k < samp.size(); ++k){
  //     ans[k] = std::lower_bound(sv.begin(), sv.end(), samp[k]) - sv.begin();
  //   }
  //   //NumericVector ansnum = ans;
  //   testdat_trans(_,i) = (ans)/nobsref;
  // }





  //NumericMatrix transformedData(originaldata.nrow(), originaldata.ncol());
  //arma::mat data_arma= as<arma::mat>(originaldata);

  //THIS CAN BE PARALLELIZED IF THERE ARE MANY VARIABLES
  arma::mat arma_test_data(testdata_arma.n_rows,testdata_arma.n_cols);
  for(unsigned int k=0; k<data_arma.n_cols;k++){
    arma::vec ref= data_arma.col(k);
    arma::vec samp= testdata_arma.col(k);

    arma::vec sv=arma::sort(samp);
    arma::vec sref=arma::sort(ref);

    //std::sort(sv.begin(), sv.end());
    arma::uvec ord = arma::sort_index(samp);
    double nobs = samp.n_elem;
    double nobsref = ref.n_elem;

    arma::vec ans(nobs);
    for (unsigned int i = 0, j = 0; i < nobs; ++i) {
      int ind=ord(i);
      double ssampi(samp[ind]);
      if(j+1>sref.size()){
      }else{
        while (sref(j) < ssampi && j < sref.size()){
          ++j;
          if(j==sref.size()) break;
        }
      }
      ans(ind) = j;     // j is the 1-based index of the lower bound
    }

    arma_test_data.col(k)=(ans)/nobsref;

  }







  /////////////////////////////////////////////////////////////////////////////////////////



  //////////////////////////////////////////////////////////////////////////////////////
  //List table_list = draw_trees(lambda, num_trees, seed, num_split_vars, num_cats );



  //dqrng::dqRNGkind("Xoroshiro128+");
  //dqrng::dqset_seed(IntegerVector::create(seed));

  //use following with binomial?
  //dqrng::xoshiro256plus rng(seed);

  //std::vector<int> lambdavec = {lambda, 1-lambda};

  //typedef boost::mt19937 RNGType;
  //boost::random::uniform_int_distribution<> sample_splitvardist(1,num_split_vars);
  //boost::variate_generator< RNGType, boost::uniform_int<> >  sample_splitvars(rng, sample_splitvardist);

  //boost::random::uniform_real_distribution<double> b_unifdist(0,1);
  //boost::variate_generator< RNGType, boost::uniform_real<> >  b_unif_point(rng, b_unifdist);



  std::random_device device;
  std::mt19937 gen(device());

  //possibly use seed?
  //// std::mt19937 gen(seed);


  std::bernoulli_distribution coin_flip(lambda);

  std::uniform_int_distribution<> distsampvar(1, num_split_vars);
  std::uniform_real_distribution<> dis_cont_unif(0, 1);




  //arma::mat arma_test_data(testdat_trans.begin(), testdat_trans.nrow(), testdat_trans.ncol(), false);


  arma::mat pred_mat_overall=arma::zeros<arma::mat>(arma_test_data.n_rows,num_cats);


  //List overall_treetables(num_trees);
  arma::vec overall_liks(num_trees);


  //overall_treetables[i]= wrap(tree_table1);
  //double templik = as<double>(treepred_output[1]);
  //overall_liks[i]= pow(lik_prod,beta_pow);



  for(int j=0; j<num_trees;j++){

    //If parallelizing, define the distributinos before this loop
    //and use lrng and the following two lines
    //dqrng::xoshiro256plus lrng(rng);      // make thread local copy of rng
    //lrng.jump(omp_get_thread_num() + 1);  // advance rng by 1 ... nthreads jumps


    //NumericVector treenodes_bin(0);
    //arma::uvec treenodes_bin(0);

    std::vector<int> treenodes_bin;


    int count_terminals = 0;
    int count_internals = 0;

    //int count_treebuild = 0;

    while(count_internals > (count_terminals -1)){

      //Also consider standard library and random header
      // std::random_device device;
      // std::mt19937 gen(device());
      // std::bernoulli_distribution coin_flip(lambda);
      // bool outcome = coin_flip(gen);


      int tempdraw = coin_flip(gen);

      //int tempdraw = rbinom(n = 1, prob = lambda,size=1);


      //int tempdraw = Rcpp::rbinom(1,lambda,1);
      //int tempdraw = R::rbinom(1,lambda);
      treenodes_bin.push_back(tempdraw);

      //Rcout << "tempdraw = " << tempdraw << ".\n" ;

      //int tempdraw = dqrng::dqsample_int(2, 1, true,lambdavec )-1;
      //need to update rng if use boost?
      //int tempdraw = bernoulli(rng, binomial::param_type(1, lambda));
      if(tempdraw==1){
        count_internals=count_internals+1;
      }else{
        count_terminals=count_terminals+1;
      }

    }//end of while loop creating parent vector treenodes_bin

    //Consider making this an armadillo vector
    //IntegerVector split_var_vec(treenodes_bin.size());
    //arma::uvec split_var_vec(treenodes_bin.size());
    std::vector<int> split_var_vec(treenodes_bin.size());

    //loop drawing splitting variables
    //REPLACE SQUARE BRACKETS WITH "( )" if using ARMADILLO vector for split_var_vec or treenodes_bin

    //if using armadillo, it might be faster to subset to split nodes
    //then use a vector of draws
    for(unsigned int i=0; i<treenodes_bin.size();i++){
      if(treenodes_bin[i]==0){
        split_var_vec[i] = -1;
      }else{
        // also consider the standard library function uniform_int_distribution
        // might need random header
        // This uses the Mersenne twister

        //Three lines below should probably be outside all the loops
        // std::random_device rd;
        // std::mt19937 engine(rd());
        // std::uniform_int_distribution<> distsampvar(1, num_split_vars);
        //
        // split_var_vec[i] = distsampvar(engine);

        split_var_vec[i] = distsampvar(gen);


        //consider using boost
        //might need to update rng
        //split_var_vec[i] <- sample_splitvars(rng);

        //or use dqrng
        //not sure if have to update the random number
        //check if the following line is written properly
        //split_var_vec[i] = dqrng::dqsample_int(num_split_vars, 1, true);

        //not sure if this returns an integer or a vector?
        //split_var_vec[i] = RcppArmadillo::sample(num_split_vars, 1,true);
        //could try
        //split_var_vec[i] = as<int>(Rcpp::sample(num_split_vars, 1,true));
        //could also try RcppArmadillo::rmultinom

      }

    }// end of for-loop drawing split variables


    //Consider making this an armadillo vector
    //NumericVector split_point_vec(treenodes_bin.size());
    //arma::vec split_point_vec(treenodes_bin.size());
    std::vector<double> split_point_vec(treenodes_bin.size());


    //loop drawing splitting points
    //REPLACE SQUARE BRACKETS WITH "( )" if using ARMADILLO vector for split_var_vec or treenodes_bin

    //if using armadillo, it might be faster to subset to split nodes
    //then use a vector of draws
    for(unsigned int i=0; i<treenodes_bin.size();i++){
      if(treenodes_bin[i]==0){
        split_point_vec[i] = -1;
      }else{


        //////////////////////////////////////////////////////////
        //following function not reccommended
        //split_point_vec[i] = std::rand();
        //////////////////////////////////////////////////////////
        ////Standard library:
        ////This should probably be outside all the loops
        ////std::random_device rd;  //Will be used to obtain a seed for the random number engine
        ////std::mt19937 gen2(rd()); //Standard mersenne_twister_engine seeded with rd()
        ////std::uniform_real_distribution<> dis_cont_unif(0, 1);

        split_point_vec[i] = dis_cont_unif(gen);

        //////////////////////////////////////////////////////////
        //from armadillo
        //split_point_vec[i] = arma::randu();

        //////////////////////////////////////////////////////////
        //probably not adviseable for paralelization
        //From Rcpp
        //split_point_vec[i] = as<double>(Rcpp::runif(1,0,1));

        //////////////////////////////////////////////////////////
        //consider using boost
        //might need to update rng
        //split_point_vec[i] <- b_unif_point(rng);

        //or use dqrng
        //not sure if have to update the random number
        //check if the following line is written properly
        //split_point_vec[i] = dqrng::dqrunif(1, 0, 1);

        //not sure if this returns an integer or a vector?





      }

    }// end of for-loop drawing split points


    //Create tree table matrix

    //NumericMatrix tree_table1(treenodes_bin.size(),5+num_cats);

    //Rcout << "Line 1037. \n";
    //arma::mat tree_table1(treenodes_bin.size(),5+num_cats);

    //initialize with zeros. Not sure if this is necessary
    arma::mat tree_table1=arma::zeros<arma::mat>(treenodes_bin.size(),5+num_cats);
    //Rcout << "Line 1040. \n";


    //tree_table1(_,2) = wrap(split_var_vec);
    //tree_table1(_,3) = wrap(split_point_vec);
    //tree_table1(_,4) = wrap(treenodes_bin);

    //It might be more efficient to make everything an armadillo object initially
    // but then would need to replace push_back etc with a different approach (but this might be more efficient anyway)
    arma::colvec split_var_vec_arma=arma::conv_to<arma::colvec>::from(split_var_vec);
    arma::colvec split_point_vec_arma(split_point_vec);
    arma::colvec treenodes_bin_arma=arma::conv_to<arma::colvec>::from(treenodes_bin);


    //Rcout << "Line 1054. \n";

    tree_table1.col(2) = split_var_vec_arma;
    tree_table1.col(3) = split_point_vec_arma;
    tree_table1.col(4) = treenodes_bin_arma;


    //Rcout << "Line 1061. j = " << j << ". \n";



    // Now start filling in left daughter and right daughter columns
    std::vector<int> rd_spaces;
    int prev_node = -1;

    for(unsigned int i=0; i<treenodes_bin.size();i++){
      //Rcout << "Line 1061. i = " << i << ". \n";
      if(prev_node==0){
        //tree_table1(rd_spaces[rd_spaces.size()-1], 1)=i;
        //Rcout << "Line 1073. j = " << j << ". \n";

        tree_table1(rd_spaces.back(), 1)=i+1;
        //Rcout << "Line 1076. j = " << j << ". \n";

        rd_spaces.pop_back();
      }
      if(treenodes_bin[i]==1){
        //Rcout << "Line 1081. j = " << j << ". \n";

        tree_table1(i,0) = i+2;
        rd_spaces.push_back(i);
        prev_node = 1;
        //Rcout << "Line 185. j = " << j << ". \n";

      }else{                  // These 2 lines unnecessary if begin with matrix of zeros
        //Rcout << "Line 1089. j = " << j << ". \n";
        tree_table1(i,0)=0 ;
        tree_table1(i,1) = 0 ;
        prev_node = 0;
        //Rcout << "Line 1093. j = " << j << ". \n";

      }
    }//
    //Rcout << "Line 1097. j = " << j << ". \n";





    //List treepred_output = get_treepreds(original_y, num_cats, alpha_pars,
    //                                     originaldata,
    //                                     treetable_list[i]  );


    //use armadillo object tree_table1

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////



    double lik_prod=1;
    double alph_prod=1;
    for(unsigned int i=0; i<alpha_pars_arma.n_elem;i++){
      alph_prod=alph_prod*tgamma(alpha_pars_arma(i));
    }
    double gam_alph_sum= tgamma(arma::sum(alpha_pars_arma));
    double alph_term=gam_alph_sum/alph_prod;

    //arma::mat arma_tree_table(treetable.begin(), treetable.nrow(), treetable.ncol(), false);
    //arma::mat arma_orig_data(originaldata.begin(), originaldata.nrow(), originaldata.ncol(), false);


    //arma::mat arma_tree(tree_data.begin(), tree_data.nrow(), tree_data.ncol(), false);
    //arma::mat testd(test_data.begin(), test_data.nrow(), test_data.ncol(), false);

    //NumericVector internal_nodes=find_internal_nodes_gs(tree_data);

    //NumericVector terminal_nodes=find_term_nodes(treetable);

    //arma::mat arma_tree(tree_table.begin(),tree_table.nrow(), tree_table.ncol(), false);

    //arma::vec colmat=arma_tree.col(4);
    //arma::uvec term_nodes=arma::find(colmat==-1);

    //arma::vec colmat=arma_tree.col(2);
    //arma::uvec term_nodes=arma::find(colmat==0);

    arma::vec colmat=tree_table1.col(4);
    arma::uvec term_nodes=arma::find(colmat==0);

    term_nodes=term_nodes+1;

    //NumericVector terminal_nodes= wrap(term_nodes);




    //arma::vec arma_terminal_nodes=Rcpp::as<arma::vec>(terminal_nodes);
    //NumericVector tree_predictions;

    //now for each internal node find the observations that belong to the terminal nodes

    //NumericVector predictions(test_data.nrow());
    //List term_obs(term_nodes.n_elem);
    if(term_nodes.n_elem==1){
      //double nodemean=tree_data(terminal_nodes[0]-1,5);				// let nodemean equal tree_data row terminal_nodes[i]^th row , 6th column. The minus 1 is because terminal nodes consists of indices starting at 1, but need indices to start at 0.
      //predictions=rep(nodemean,test_data.nrow());
      //Rcout << "Line 67 .\n";

      //IntegerVector temp_obsvec = seq_len(test_data.nrow())-1;
      //term_obs[0]= temp_obsvec;
      double denom_temp= orig_y_arma.n_elem+arma::sum(alpha_pars_arma);

      double num_prod=1;
      double num_sum=0;
      //Rcout << "Line 129.\n";

      for(int k=0; k<num_cats; k++){
        //assuming categories of y are from 1 to num_cats
        arma::uvec cat_inds= arma::find(orig_y_arma==k+1);
        double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
        tree_table1(0,5+k)= m_plus_alph/denom_temp ;

        //for likelihood calculation
        num_prod=num_prod*tgamma(m_plus_alph);
        num_sum=num_sum +m_plus_alph ;
      }

      lik_prod= alph_term*num_prod/tgamma(num_sum);

    }
    else{
      for(unsigned int i=0;i<term_nodes.n_elem;i++){
        //arma::mat subdata=testd;
        int curr_term=term_nodes(i);

        int row_index;
        int term_node=term_nodes(i);
        //Rcout << "Line 152.\n";


        //WHAT IS THE PURPOSE OF THIS IF-STATEMENT?
        //Why should the ro index be different for a right daughter?
        //Why not just initialize row_index to any number not equal to 1 (e.g. 0)?
        row_index=0;

        // if(curr_term % 2==0){
        //   //term node is left daughter
        //   row_index=terminal_nodes[i];
        // }else{
        //   //term node is right daughter
        //   row_index=terminal_nodes[i]-1;
        // }




        //save the left and right node data into arma uvec

        //CHECK THAT THIS REFERS TO THE CORRECT COLUMNS
        //arma::vec left_nodes=arma_tree.col(0);
        //arma::vec right_nodes=arma_tree.col(1);

        arma::vec left_nodes=tree_table1.col(0);
        arma::vec right_nodes=tree_table1.col(1);



        arma::mat node_split_mat;
        node_split_mat.set_size(0,3);
        //Rcout << "Line 182. i = " << i << " .\n";

        while(row_index!=1){
          //for each terminal node work backwards and see if the parent node was a left or right node
          //append split info to a matrix
          int rd=0;
          arma::uvec parent_node=arma::find(left_nodes == term_node);

          if(parent_node.size()==0){
            parent_node=arma::find(right_nodes == term_node);
            rd=1;
          }

          //want to cout parent node and append to node_split_mat

          node_split_mat.insert_rows(0,1);

          //CHECK THAT COLUMNS OF TREETABLE ARE CORRECT
          //node_split_mat(0,0)=treetable(parent_node[0],2);
          //node_split_mat(0,1)=treetable(parent_node[0],3);

          //node_split_mat(0,0)=arma_tree_table(parent_node[0],3);
          //node_split_mat(0,1)=arma_tree_table(parent_node[0],4);

          node_split_mat(0,0)=tree_table1(parent_node(0),2);
          node_split_mat(0,1)=tree_table1(parent_node(0),3);

          node_split_mat(0,2)=rd;
          row_index=parent_node(0)+1;
          term_node=parent_node(0)+1;
        }

        //once we have the split info, loop through rows and find the subset indexes for that terminal node!
        //then fill in the predicted value for that tree
        //double prediction = tree_data(term_node,5);
        arma::uvec pred_indices;
        int split= node_split_mat(0,0)-1;

        //Rcout << "Line 224.\n";
        //Rcout << "split = " << split << ".\n";
        //arma::vec tempvec = testd.col(split);
        arma::vec tempvec = arma_orig_data.col(split);
        //Rcout << "Line 227.\n";


        double temp_split = node_split_mat(0,1);

        if(node_split_mat(0,2)==0){
          pred_indices = arma::find(tempvec <= temp_split);
        }else{
          pred_indices = arma::find(tempvec > temp_split);
        }
        //Rcout << "Line 236.\n";

        arma::uvec temp_pred_indices;

        //arma::vec data_subset = testd.col(split);
        arma::vec data_subset = arma_orig_data.col(split);

        data_subset=data_subset.elem(pred_indices);

        //now loop through each row of node_split_mat
        int n=node_split_mat.n_rows;
        //Rcout << "Line 174. i = " << i << ". n = " << n << ".\n";
        //Rcout << "Line 248.\n";

        for(int j=1;j<n;j++){
          int curr_sv=node_split_mat(j,0);
          double split_p = node_split_mat(j,1);

          //data_subset = testd.col(curr_sv-1);
          //Rcout << "Line 255.\n";
          //Rcout << "curr_sv = " << curr_sv << ".\n";
          data_subset = arma_orig_data.col(curr_sv-1);
          //Rcout << "Line 258.\n";

          data_subset=data_subset.elem(pred_indices);

          if(node_split_mat(j,2)==0){
            //split is to the left
            temp_pred_indices=arma::find(data_subset <= split_p);
          }else{
            //split is to the right
            temp_pred_indices=arma::find(data_subset > split_p);
          }
          pred_indices=pred_indices.elem(temp_pred_indices);

          if(pred_indices.size()==0){
            continue;
          }

        }
        //Rcout << "Line 199. i = " << i <<  ".\n";

        //double nodemean=tree_data(terminal_nodes[i]-1,5);
        //IntegerVector predind=as<IntegerVector>(wrap(pred_indices));
        //predictions[predind]= nodemean;
        //term_obs[i]=predind;

        double denom_temp= pred_indices.n_elem+arma::sum(alpha_pars_arma);
        //Rcout << "Line 207. predind = " << predind <<  ".\n";
        //Rcout << "Line 207. denom_temp = " << denom_temp <<  ".\n";
        // << "Line 207. term_node = " << term_node <<  ".\n";

        double num_prod=1;
        double num_sum=0;

        for(int k=0; k<num_cats; k++){
          //assuming categories of y are from 1 to num_cats
          arma::uvec cat_inds= arma::find(orig_y_arma(pred_indices)==k+1);
          double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);

          tree_table1(curr_term-1,5+k)= m_plus_alph/denom_temp ;

          num_prod=num_prod*tgamma(m_plus_alph);
          num_sum=num_sum +m_plus_alph ;
        }


        lik_prod= lik_prod*alph_term*num_prod/tgamma(num_sum);
        //Rcout << "Line 297.\n";


      }
      //Rcout << "Line 301.\n";

    }
    //List ret(1);
    //ret[0] = term_obs;

    //ret[0] = terminal_nodes;
    //ret[1] = term_obs;
    //ret[2] = predictions;
    //return(term_obs);
    //Rcout << "Line 309";

    //return(wrap(arma_tree_table));

    //List ret(2);
    //ret[0]=wrap(arma_tree_table);
    //ret[1]=lik_prod;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////






    //overall_treetables[j]= wrap(tree_table1);


    //double templik = as<double>(treepred_output[1]);

    double templik = pow(lik_prod,beta_par);
    overall_liks(j)= templik;






    //arma::mat arma_tree_table(treetable.begin(), treetable.nrow(), treetable.ncol(), false);
    //arma::mat arma_test_data(testdata.begin(), testdata.nrow(), testdata.ncol(), false);


    //arma::mat arma_tree(tree_data.begin(), tree_data.nrow(), tree_data.ncol(), false);
    //arma::mat testd(test_data.begin(), test_data.nrow(), test_data.ncol(), false);

    //NumericVector internal_nodes=find_internal_nodes_gs(tree_data);

    //NumericVector terminal_nodes=find_term_nodes(treetable);
    //arma::vec arma_terminal_nodes=Rcpp::as<arma::vec>(terminal_nodes);
    //NumericVector tree_predictions;

    //now for each internal node find the observations that belong to the terminal nodes

    //NumericVector predictions(test_data.nrow());

    arma::mat pred_mat(testdata_arma.n_rows,num_cats);
    //arma::vec filled_in(testdata.nrow());


    //List term_obs(terminal_nodes.size());
    if(term_nodes.size()==1){

      //Rcout << "Line 422. \n";


      pred_mat=repmat(tree_table1(0,arma::span(5,5+num_cats-1)),testdata_arma.n_rows,1);


      //Rcout << "Line 424. \n";


      // for(int k=0; k<num_cats; k++){
      // pred_mat(_,k)=rep(treetable(0,5+k),testdata.nrow());
      // }
      //double nodemean=tree_data(terminal_nodes[0]-1,5);				// let nodemean equal tree_data row terminal_nodes[i]^th row , 6th column. The minus 1 is because terminal nodes consists of indices starting at 1, but need indices to start at 0.
      //predictions=rep(nodemean,test_data.nrow());
      //Rcout << "Line 67 .\n";

      //IntegerVector temp_obsvec = seq_len(test_data.nrow())-1;
      //term_obs[0]= temp_obsvec;
      // double denom_temp= orig_y_arma.n_elem+arma::sum(alpha_pars_arma);
      //
      // double num_prod=1;
      // double num_sum=0;

      // for(int k=0; k<num_cats; k++){
      //   //assuming categories of y are from 1 to num_cats
      //   arma::uvec cat_inds= arma::find(orig_y_arma==k+1);
      //   double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
      //   arma_tree_table(0,5+k)= m_plus_alph/denom_temp ;
      //
      //   //for likelihood calculation
      //   num_prod=num_prod*tgamma(m_plus_alph);
      //   num_sum=num_sum +m_plus_alph ;
      // }
      //
      // lik_prod= alph_term*num_prod/tgamma(num_sum);
      //
    }
    else{
      for(unsigned int i=0;i<term_nodes.size();i++){
        //arma::mat subdata=testd;
        int curr_term=term_nodes(i);

        int row_index;
        int term_node=term_nodes(i);


        //WHAT IS THE PURPOSE OF THIS IF-STATEMENT?
        //Why should the ro index be different for a right daughter?
        //Why not just initialize row_index to any number not equal to 1 (e.g. 0)?
        row_index=0;

        // if(curr_term % 2==0){
        //   //term node is left daughter
        //   row_index=terminal_nodes[i];
        // }else{
        //   //term node is right daughter
        //   row_index=terminal_nodes[i]-1;
        // }








        //save the left and right node data into arma uvec

        //CHECK THAT THIS REFERS TO THE CORRECT COLUMNS
        //arma::vec left_nodes=arma_tree.col(0);
        //arma::vec right_nodes=arma_tree.col(1);

        arma::vec left_nodes=tree_table1.col(0);
        arma::vec right_nodes=tree_table1.col(1);



        arma::mat node_split_mat;
        node_split_mat.set_size(0,3);
        //Rcout << "Line 124. i = " << i << " .\n";

        while(row_index!=1){
          //for each terminal node work backwards and see if the parent node was a left or right node
          //append split info to a matrix
          int rd=0;
          arma::uvec parent_node=arma::find(left_nodes == term_node);

          if(parent_node.size()==0){
            parent_node=arma::find(right_nodes == term_node);
            rd=1;
          }

          //want to cout parent node and append to node_split_mat

          node_split_mat.insert_rows(0,1);

          //CHECK THAT COLUMNS OF TREETABLE ARE CORRECT
          //node_split_mat(0,0)=treetable(parent_node[0],2);
          //node_split_mat(0,1)=treetable(parent_node[0],3);

          //node_split_mat(0,0)=arma_tree_table(parent_node[0],3);
          //node_split_mat(0,1)=arma_tree_table(parent_node[0],4);

          node_split_mat(0,0)=tree_table1(parent_node(0),2);
          node_split_mat(0,1)=tree_table1(parent_node(0),3);

          node_split_mat(0,2)=rd;
          row_index=parent_node(0)+1;
          term_node=parent_node(0)+1;
        }

        //once we have the split info, loop through rows and find the subset indexes for that terminal node!
        //then fill in the predicted value for that tree
        //double prediction = tree_data(term_node,5);
        arma::uvec pred_indices;
        int split= node_split_mat(0,0)-1;

        //arma::vec tempvec = testd.col(split);
        arma::vec tempvec = arma_test_data.col(split);


        double temp_split = node_split_mat(0,1);

        if(node_split_mat(0,2)==0){
          pred_indices = arma::find(tempvec <= temp_split);
        }else{
          pred_indices = arma::find(tempvec > temp_split);
        }

        arma::uvec temp_pred_indices;

        //arma::vec data_subset = testd.col(split);
        arma::vec data_subset = arma_test_data.col(split);

        data_subset=data_subset.elem(pred_indices);

        //now loop through each row of node_split_mat
        int n=node_split_mat.n_rows;
        //Rcout << "Line 174. i = " << i << ". n = " << n << ".\n";
        //Rcout << "Line 174. node_split_mat= " << node_split_mat << ". n = " << n << ".\n";


        for(int j=1;j<n;j++){
          int curr_sv=node_split_mat(j,0);
          double split_p = node_split_mat(j,1);

          //data_subset = testd.col(curr_sv-1);
          data_subset = arma_test_data.col(curr_sv-1);

          data_subset=data_subset.elem(pred_indices);

          if(node_split_mat(j,2)==0){
            //split is to the left
            temp_pred_indices=arma::find(data_subset <= split_p);
          }else{
            //split is to the right
            temp_pred_indices=arma::find(data_subset > split_p);
          }
          pred_indices=pred_indices.elem(temp_pred_indices);

          if(pred_indices.size()==0){
            continue;
          }

        }
        //Rcout << "Line 199. i = " << i <<  ".\n";

        //double nodemean=tree_data(terminal_nodes[i]-1,5);
        //IntegerVector predind=as<IntegerVector>(wrap(pred_indices));
        //predictions[predind]= nodemean;
        //term_obs[i]=predind;

        //Rcout << "Line 635. \n";
        //Rcout << "pred_indices = " << pred_indices << ".\n";

        //pred_mat.rows(pred_indices)=arma::repmat(arma_tree_table(curr_term-1,arma::span(5,5+num_cats-1)),pred_indices.n_elem,1);
        pred_mat.each_row(pred_indices)=tree_table1(curr_term-1,arma::span(5,4+num_cats));



        //Rcout << "Line 588. \n";

        // for(int k=0; k<num_cats; k++){
        //   pred_mat(predind,k)=rep(treetable(curr_term-1,5+k),predind.size());
        // }


        // double denom_temp= pred_indices.n_elem+arma::sum(alpha_pars_arma);
        // //Rcout << "Line 207. predind = " << predind <<  ".\n";
        // //Rcout << "Line 207. denom_temp = " << denom_temp <<  ".\n";
        // // << "Line 207. term_node = " << term_node <<  ".\n";
        //
        // double num_prod=1;
        // double num_sum=0;
        //
        // for(int k=0; k<num_cats; k++){
        //   //assuming categories of y are from 1 to num_cats
        //   arma::uvec cat_inds= arma::find(orig_y_arma(pred_indices)==k+1);
        //   double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
        //
        //   arma_tree_table(curr_term-1,5+k)= m_plus_alph/denom_temp ;
        //
        //   num_prod=num_prod*tgamma(m_plus_alph);
        //   num_sum=num_sum +m_plus_alph ;
        // }


        //lik_prod= lik_prod*alph_term*num_prod/tgamma(num_sum);


      }
    }






    //THIS SHOULD BE DIFFERENT IF THE CODE IS TO BE PARALLELIZED
    //EACH THREAD SHOULD OUTPUT ITS OWN MATRIX AND SUM OF LIKELIHOODS
    //THEN ADD THE MATRICES TOGETHER AND DIVIDE BY THE TOTAL SUM OF LIKELIHOODS
    //OR JUST SAVE ALL MATRICES TO ONE LIST

    pred_mat_overall = pred_mat_overall + templik*pred_mat;




    //arma::mat treeprob_output = get_test_probs(weights, num_cats,
    //                                           testdata,
    //                                           treetable_list[i]  );

    //Rcout << "Line 688. i== " << i << ". \n";

    //double weighttemp = weights[i];
    //Rcout << "Line 691. i== " << i << ". \n";

    //pred_mat_overall = pred_mat_overall + weighttemp*treeprob_output;



  }//end of loop over all trees




  ///////////////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////////



  double sumlik_total= arma::sum(overall_liks);
  pred_mat_overall=pred_mat_overall*(1/sumlik_total);
  //Rcout << "Line 1141 . \n";
  //Rcout << "Line 1146 . \n";

  return(wrap(pred_mat_overall));

}
//######################################################################################################################//

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(dqrng, BH, sitmo)]]
#include <xoshiro.h>
#include <dqrng_distribution.h>
//#include <dqrng.h>

// [[Rcpp::plugins(openmp)]]
#include <omp.h>

//' @title Parallel Safe-Bayesian Random Forest
//'
//' @description A parallelized implementation of the Safe-Bayesian Random Forest described by Quadrianto and Ghahramani (2015)
//' @param lambda A real number between 0 and 1 that determines the splitting probability in the prior (which is used as the importance sampler of tree models). Quadrianto and Ghahramani (2015) recommend a value less than 0.5 .
//' @param num_trees The number of trees to be sampled.
//' @param seed The seed for random number generation.
//' @param num_cats The number of possible values for the outcome variable.
//' @param y The training data vector of outcomes. This must be a vector of integers between 1 and num_cats.
//' @param original_datamat The original training data. Currently all variables must be continuous. The training data does not need to be transformed before being entered to this function.
//' @param alpha_parameters Vector of prior parameters.
//' @param beta_par The power to which the likelihood is to be raised. For BMA, set beta_par=1.
//' @param original_datamat The original test data. This matrix must have the same number of columns (variables) as the training data. Currently all variables must be continuous. The test data does not need to be transformed before being entered to this function.
//' @param ncores The number of cores to be used in parallelization.
//' @return A matrix of probabilities with the number of rows equl to the number of test observations and the number of columns equal to the number of possible outcome categories.
//' @export
// [[Rcpp::export]]
NumericMatrix sBayesRF_onefunc_parallel(double lambda, int num_trees,
                                        int seed, int num_cats,
                                        NumericVector y, NumericMatrix original_datamat,
                                        NumericVector alpha_parameters, double beta_par,
                                        NumericMatrix test_datamat, int ncores){

  int num_split_vars= original_datamat.ncol();
  arma::mat data_arma= as<arma::mat>(original_datamat);
  arma::mat testdata_arma= as<arma::mat>(test_datamat);
  arma::vec orig_y_arma= as<arma::vec>(y);
  arma::vec alpha_pars_arma= as<arma::vec>(alpha_parameters);

  ///////////////////////
  //NumericMatrix Data_transformed = cpptrans_cdf(original_datamat);
  // NumericMatrix Data_transformed(original_datamat.nrow(), original_datamat.ncol());
  // for(int i=0; i<original_datamat.ncol();i++){
  //   NumericVector samp= original_datamat(_,i);
  //   NumericVector sv(clone(samp));
  //   std::sort(sv.begin(), sv.end());
  //   double nobs = samp.size();
  //   NumericVector ans(nobs);
  //   for (int k = 0; k < samp.size(); ++k)
  //     ans[k] = std::lower_bound(sv.begin(), sv.end(), samp[k]) - sv.begin();
  //   //NumericVector ansnum = ans;
  //   Data_transformed(_,i) = (ans+1)/nobs;
  // }



  //arma::mat arma_orig_data(Data_transformed.begin(), Data_transformed.nrow(), Data_transformed.ncol(), false);



  //NumericMatrix transformedData(originaldata.nrow(), originaldata.ncol());

  //THIS CAN BE PARALLELIZED IF THERE ARE MANY VARIABLES
  arma::mat arma_orig_data(data_arma.n_rows,data_arma.n_cols);
  for(unsigned int k=0; k<data_arma.n_cols;k++){
    arma::vec samp= data_arma.col(k);
    arma::vec sv=arma::sort(samp);
    //std::sort(sv.begin(), sv.end());
    arma::uvec ord = arma::sort_index(samp);
    double nobs = samp.n_elem;
    arma::vec ans(nobs);
    for (unsigned int i = 0, j = 0; i < nobs; ++i) {
      int ind=ord(i);
      double ssampi(samp[ind]);
      while (sv(j) < ssampi && j < sv.size()) ++j;
      ans(ind) = j;     // j is the 1-based index of the lower bound
    }
    arma_orig_data.col(k)=(ans+1)/nobs;
  }





  /////////////////////////////////////
  // NumericMatrix testdat_trans = cpptrans_cdf_test(original_datamat,test_datamat);
  // //NumericMatrix testdat_trans(test_datamat.nrow(), test_datamat.ncol());
  // for(int i=0; i<test_datamat.ncol();i++){
  //   NumericVector samp= test_datamat(_,i);
  //   NumericVector svtest = original_datamat(_,i);
  //   NumericVector sv(clone(svtest));
  //   std::sort(sv.begin(), sv.end());
  //   double nobs = samp.size();
  //   NumericVector ans(nobs);
  //   double nobsref = svtest.size();
  //   for (int k = 0; k < samp.size(); ++k){
  //     ans[k] = std::lower_bound(sv.begin(), sv.end(), samp[k]) - sv.begin();
  //   }
  //   //NumericVector ansnum = ans;
  //   testdat_trans(_,i) = (ans)/nobsref;
  // }





  //NumericMatrix transformedData(originaldata.nrow(), originaldata.ncol());
  //arma::mat data_arma= as<arma::mat>(originaldata);

  //THIS CAN BE PARALLELIZED IF THERE ARE MANY VARIABLES
  arma::mat arma_test_data(testdata_arma.n_rows,testdata_arma.n_cols);
  for(unsigned int k=0; k<data_arma.n_cols;k++){
    arma::vec ref= data_arma.col(k);
    arma::vec samp= testdata_arma.col(k);

    arma::vec sv=arma::sort(samp);
    arma::vec sref=arma::sort(ref);

    //std::sort(sv.begin(), sv.end());
    arma::uvec ord = arma::sort_index(samp);
    double nobs = samp.n_elem;
    double nobsref = ref.n_elem;

    arma::vec ans(nobs);
    for (unsigned int i = 0, j = 0; i < nobs; ++i) {
      int ind=ord(i);
      double ssampi(samp[ind]);
      if(j+1>sref.size()){
      }else{
        while (sref(j) < ssampi && j < sref.size()){
          ++j;
          if(j==sref.size()) break;
        }
      }
      ans(ind) = j;     // j is the 1-based index of the lower bound
    }

    arma_test_data.col(k)=(ans)/nobsref;

  }







  /////////////////////////////////////////////////////////////////////////////////////////



  //////////////////////////////////////////////////////////////////////////////////////
  //List table_list = draw_trees(lambda, num_trees, seed, num_split_vars, num_cats );



  //dqrng::dqRNGkind("Xoroshiro128+");
  //dqrng::dqset_seed(IntegerVector::create(seed));

  //use following with binomial?
  //dqrng::xoshiro256plus rng(seed);

  std::vector<double> lambdavec = {lambda, 1-lambda};

  //typedef boost::mt19937 RNGType;
  //boost::random::uniform_int_distribution<> sample_splitvardist(1,num_split_vars);
  //boost::variate_generator< RNGType, boost::uniform_int<> >  sample_splitvars(rng, sample_splitvardist);

  //boost::random::uniform_real_distribution<double> b_unifdist(0,1);
  //boost::variate_generator< RNGType, boost::uniform_real<> >  b_unif_point(rng, b_unifdist);



  std::random_device device;
  //std::mt19937 gen(device());

  //possibly use seed?
  //// std::mt19937 gen(seed);

  dqrng::xoshiro256plus gen(device());              // properly seeded rng

  //dqrng::xoshiro256plus gen(seed);              // properly seeded rng




  std::bernoulli_distribution coin_flip(lambda);

  std::uniform_int_distribution<> distsampvar(1, num_split_vars);
  std::uniform_real_distribution<> dis_cont_unif(0, 1);


  //dqrng::uniform_distribution dis_cont_unif(0.0, 1.0); // Uniform distribution [0,1)

  //Following three functions can't be used in parallel
  //dqrng::dqsample_int coin_flip2(2, 1, true,lambdavec );
  //dqrng::dqsample_int distsampvar(num_split_vars, 1, true);
  //dqrng::dqrunif dis_cont_unif(1, 0, 1);



  //arma::mat arma_test_data(testdat_trans.begin(), testdat_trans.nrow(), testdat_trans.ncol(), false);


  arma::mat pred_mat_overall=arma::zeros<arma::mat>(arma_test_data.n_rows,num_cats);


  arma::field<arma::mat> overall_treetables(num_trees);
  arma::vec overall_liks(num_trees);


  //overall_treetables[i]= wrap(tree_table1);
  //double templik = as<double>(treepred_output[1]);
  //overall_liks[i]= pow(lik_prod,beta_pow);


#pragma omp parallel num_threads(ncores)
{//start of pragma omp code
  dqrng::xoshiro256plus lgen(gen);      // make thread local copy of rng
  lgen.jump(omp_get_thread_num() + 1);  // advance rng by 1 ... ncores jumps

#pragma omp for
  for(int j=0; j<num_trees;j++){

    //If parallelizing, define the distributinos before this loop
    //and use lrng and the following two lines
    //dqrng::xoshiro256plus lrng(rng);      // make thread local copy of rng
    //lrng.jump(omp_get_thread_num() + 1);  // advance rng by 1 ... nthreads jumps


    //NumericVector treenodes_bin(0);
    //arma::uvec treenodes_bin(0);

    std::vector<int> treenodes_bin;


    int count_terminals = 0;
    int count_internals = 0;

    //int count_treebuild = 0;

    while(count_internals > (count_terminals -1)){

      //Also consider standard library and random header
      // std::random_device device;
      // std::mt19937 gen(device());
      // std::bernoulli_distribution coin_flip(lambda);
      // bool outcome = coin_flip(gen);


      int tempdraw = coin_flip(lgen);

      //int tempdraw = rbinom(n = 1, prob = lambda,size=1);


      //int tempdraw = Rcpp::rbinom(1,lambda,1);
      //int tempdraw = R::rbinom(1,lambda);

      //Rcout << "tempdraw = " << tempdraw << ".\n" ;

      //int tempdraw = coin_flip2(lgen)-1;

      //int tempdraw = dqrng::dqsample_int(2, 1, true,lambdavec )-1;


      //need to update rng if use boost?
      //int tempdraw = bernoulli(rng, binomial::param_type(1, lambda));

      treenodes_bin.push_back(tempdraw);


      if(tempdraw==1){
        count_internals=count_internals+1;
      }else{
        count_terminals=count_terminals+1;
      }

    }//end of while loop creating parent vector treenodes_bin

    //Consider making this an armadillo vector
    //IntegerVector split_var_vec(treenodes_bin.size());
    //arma::uvec split_var_vec(treenodes_bin.size());
    std::vector<int> split_var_vec(treenodes_bin.size());

    //loop drawing splitting variables
    //REPLACE SQUARE BRACKETS WITH "( )" if using ARMADILLO vector for split_var_vec or treenodes_bin

    //if using armadillo, it might be faster to subset to split nodes
    //then use a vector of draws
    for(unsigned int i=0; i<treenodes_bin.size();i++){
      if(treenodes_bin[i]==0){
        split_var_vec[i] = -1;
      }else{
        // also consider the standard library function uniform_int_distribution
        // might need random header
        // This uses the Mersenne twister

        //Three lines below should probably be outside all the loops
        // std::random_device rd;
        // std::mt19937 engine(rd());
        // std::uniform_int_distribution<> distsampvar(1, num_split_vars);
        //
        // split_var_vec[i] = distsampvar(engine);

        split_var_vec[i] = distsampvar(lgen);


        //consider using boost
        //might need to update rng
        //split_var_vec[i] <- sample_splitvars(rng);

        //or use dqrng
        //not sure if have to update the random number
        //check if the following line is written properly
        //split_var_vec[i] = dqrng::dqsample_int(num_split_vars, 1, true);

        //not sure if this returns an integer or a vector?
        //split_var_vec[i] = RcppArmadillo::sample(num_split_vars, 1,true);
        //could try
        //split_var_vec[i] = as<int>(Rcpp::sample(num_split_vars, 1,true));
        //could also try RcppArmadillo::rmultinom

      }

    }// end of for-loop drawing split variables


    //Consider making this an armadillo vector
    //NumericVector split_point_vec(treenodes_bin.size());
    //arma::vec split_point_vec(treenodes_bin.size());
    std::vector<double> split_point_vec(treenodes_bin.size());


    //loop drawing splitting points
    //REPLACE SQUARE BRACKETS WITH "( )" if using ARMADILLO vector for split_var_vec or treenodes_bin

    //if using armadillo, it might be faster to subset to split nodes
    //then use a vector of draws
    for(unsigned int i=0; i<treenodes_bin.size();i++){
      if(treenodes_bin[i]==0){
        split_point_vec[i] = -1;
      }else{


        //////////////////////////////////////////////////////////
        //following function not reccommended
        //split_point_vec[i] = std::rand();
        //////////////////////////////////////////////////////////
        ////Standard library:
        ////This should probably be outside all the loops
        ////std::random_device rd;  //Will be used to obtain a seed for the random number engine
        ////std::mt19937 gen2(rd()); //Standard mersenne_twister_engine seeded with rd()
        ////std::uniform_real_distribution<> dis_cont_unif(0, 1);

        split_point_vec[i] = dis_cont_unif(lgen);

        //////////////////////////////////////////////////////////
        //from armadillo
        //split_point_vec[i] = arma::randu();

        //////////////////////////////////////////////////////////
        //probably not adviseable for paralelization
        //From Rcpp
        //split_point_vec[i] = as<double>(Rcpp::runif(1,0,1));

        //////////////////////////////////////////////////////////
        //consider using boost
        //might need to update rng
        //split_point_vec[i] <- b_unif_point(rng);

        //or use dqrng
        //not sure if have to update the random number
        //check if the following line is written properly
        //split_point_vec[i] = dqrng::dqrunif(1, 0, 1);

        //not sure if this returns an integer or a vector?





      }

    }// end of for-loop drawing split points


    //Create tree table matrix

    //NumericMatrix tree_table1(treenodes_bin.size(),5+num_cats);

    //Rcout << "Line 1037. \n";
    //arma::mat tree_table1(treenodes_bin.size(),5+num_cats);

    //initialize with zeros. Not sure if this is necessary
    arma::mat tree_table1=arma::zeros<arma::mat>(treenodes_bin.size(),5+num_cats);
    //Rcout << "Line 1040. \n";


    //tree_table1(_,2) = wrap(split_var_vec);
    //tree_table1(_,3) = wrap(split_point_vec);
    //tree_table1(_,4) = wrap(treenodes_bin);

    //It might be more efficient to make everything an armadillo object initially
    // but then would need to replace push_back etc with a different approach (but this might be more efficient anyway)
    arma::colvec split_var_vec_arma=arma::conv_to<arma::colvec>::from(split_var_vec);
    arma::colvec split_point_vec_arma(split_point_vec);
    arma::colvec treenodes_bin_arma=arma::conv_to<arma::colvec>::from(treenodes_bin);


    //Rcout << "Line 1054. \n";

    tree_table1.col(2) = split_var_vec_arma;
    tree_table1.col(3) = split_point_vec_arma;
    tree_table1.col(4) = treenodes_bin_arma;


    //Rcout << "Line 1061. j = " << j << ". \n";



    // Now start filling in left daughter and right daughter columns
    std::vector<int> rd_spaces;
    int prev_node = -1;

    for(unsigned int i=0; i<treenodes_bin.size();i++){
      //Rcout << "Line 1061. i = " << i << ". \n";
      if(prev_node==0){
        //tree_table1(rd_spaces[rd_spaces.size()-1], 1)=i;
        //Rcout << "Line 1073. j = " << j << ". \n";

        tree_table1(rd_spaces.back(), 1)=i+1;
        //Rcout << "Line 1076. j = " << j << ". \n";

        rd_spaces.pop_back();
      }
      if(treenodes_bin[i]==1){
        //Rcout << "Line 1081. j = " << j << ". \n";

        tree_table1(i,0) = i+2;
        rd_spaces.push_back(i);
        prev_node = 1;
        //Rcout << "Line 185. j = " << j << ". \n";

      }else{                  // These 2 lines unnecessary if begin with matrix of zeros
        //Rcout << "Line 1089. j = " << j << ". \n";
        tree_table1(i,0)=0 ;
        tree_table1(i,1) = 0 ;
        prev_node = 0;
        //Rcout << "Line 1093. j = " << j << ". \n";

      }
    }//
    //Rcout << "Line 1097. j = " << j << ". \n";





    //List treepred_output = get_treepreds(original_y, num_cats, alpha_pars,
    //                                     originaldata,
    //                                     treetable_list[i]  );


    //use armadillo object tree_table1

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////



    double lik_prod=1;
    double alph_prod=1;
    for(unsigned int i=0; i<alpha_pars_arma.n_elem;i++){
      alph_prod=alph_prod*tgamma(alpha_pars_arma(i));
    }
    double gam_alph_sum= tgamma(arma::sum(alpha_pars_arma));
    double alph_term=gam_alph_sum/alph_prod;

    //arma::mat arma_tree_table(treetable.begin(), treetable.nrow(), treetable.ncol(), false);
    //arma::mat arma_orig_data(originaldata.begin(), originaldata.nrow(), originaldata.ncol(), false);


    //arma::mat arma_tree(tree_data.begin(), tree_data.nrow(), tree_data.ncol(), false);
    //arma::mat testd(test_data.begin(), test_data.nrow(), test_data.ncol(), false);

    //NumericVector internal_nodes=find_internal_nodes_gs(tree_data);

    //NumericVector terminal_nodes=find_term_nodes(treetable);

    //arma::mat arma_tree(tree_table.begin(),tree_table.nrow(), tree_table.ncol(), false);

    //arma::vec colmat=arma_tree.col(4);
    //arma::uvec term_nodes=arma::find(colmat==-1);

    //arma::vec colmat=arma_tree.col(2);
    //arma::uvec term_nodes=arma::find(colmat==0);

    arma::vec colmat=tree_table1.col(4);
    arma::uvec term_nodes=arma::find(colmat==0);

    term_nodes=term_nodes+1;

    //NumericVector terminal_nodes= wrap(term_nodes);




    //arma::vec arma_terminal_nodes=Rcpp::as<arma::vec>(terminal_nodes);
    //NumericVector tree_predictions;

    //now for each internal node find the observations that belong to the terminal nodes

    //NumericVector predictions(test_data.nrow());
    //List term_obs(term_nodes.n_elem);
    if(term_nodes.n_elem==1){
      //double nodemean=tree_data(terminal_nodes[0]-1,5);				// let nodemean equal tree_data row terminal_nodes[i]^th row , 6th column. The minus 1 is because terminal nodes consists of indices starting at 1, but need indices to start at 0.
      //predictions=rep(nodemean,test_data.nrow());
      //Rcout << "Line 67 .\n";

      //IntegerVector temp_obsvec = seq_len(test_data.nrow())-1;
      //term_obs[0]= temp_obsvec;
      double denom_temp= orig_y_arma.n_elem+arma::sum(alpha_pars_arma);

      double num_prod=1;
      double num_sum=0;
      //Rcout << "Line 129.\n";

      for(int k=0; k<num_cats; k++){
        //assuming categories of y are from 1 to num_cats
        arma::uvec cat_inds= arma::find(orig_y_arma==k+1);
        double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
        tree_table1(0,5+k)= m_plus_alph/denom_temp ;

        //for likelihood calculation
        num_prod=num_prod*tgamma(m_plus_alph);
        num_sum=num_sum +m_plus_alph ;
      }

      lik_prod= alph_term*num_prod/tgamma(num_sum);

    }
    else{
      for(unsigned int i=0;i<term_nodes.n_elem;i++){
        //arma::mat subdata=testd;
        int curr_term=term_nodes(i);

        int row_index;
        int term_node=term_nodes(i);
        //Rcout << "Line 152.\n";


        //WHAT IS THE PURPOSE OF THIS IF-STATEMENT?
        //Why should the ro index be different for a right daughter?
        //Why not just initialize row_index to any number not equal to 1 (e.g. 0)?
        row_index=0;

        // if(curr_term % 2==0){
        //   //term node is left daughter
        //   row_index=terminal_nodes[i];
        // }else{
        //   //term node is right daughter
        //   row_index=terminal_nodes[i]-1;
        // }




        //save the left and right node data into arma uvec

        //CHECK THAT THIS REFERS TO THE CORRECT COLUMNS
        //arma::vec left_nodes=arma_tree.col(0);
        //arma::vec right_nodes=arma_tree.col(1);

        arma::vec left_nodes=tree_table1.col(0);
        arma::vec right_nodes=tree_table1.col(1);



        arma::mat node_split_mat;
        node_split_mat.set_size(0,3);
        //Rcout << "Line 182. i = " << i << " .\n";

        while(row_index!=1){
          //for each terminal node work backwards and see if the parent node was a left or right node
          //append split info to a matrix
          int rd=0;
          arma::uvec parent_node=arma::find(left_nodes == term_node);

          if(parent_node.size()==0){
            parent_node=arma::find(right_nodes == term_node);
            rd=1;
          }

          //want to cout parent node and append to node_split_mat

          node_split_mat.insert_rows(0,1);

          //CHECK THAT COLUMNS OF TREETABLE ARE CORRECT
          //node_split_mat(0,0)=treetable(parent_node[0],2);
          //node_split_mat(0,1)=treetable(parent_node[0],3);

          //node_split_mat(0,0)=arma_tree_table(parent_node[0],3);
          //node_split_mat(0,1)=arma_tree_table(parent_node[0],4);

          node_split_mat(0,0)=tree_table1(parent_node(0),2);
          node_split_mat(0,1)=tree_table1(parent_node(0),3);

          node_split_mat(0,2)=rd;
          row_index=parent_node(0)+1;
          term_node=parent_node(0)+1;
        }

        //once we have the split info, loop through rows and find the subset indexes for that terminal node!
        //then fill in the predicted value for that tree
        //double prediction = tree_data(term_node,5);
        arma::uvec pred_indices;
        int split= node_split_mat(0,0)-1;

        //Rcout << "Line 224.\n";
        //Rcout << "split = " << split << ".\n";
        //arma::vec tempvec = testd.col(split);
        arma::vec tempvec = arma_orig_data.col(split);
        //Rcout << "Line 227.\n";


        double temp_split = node_split_mat(0,1);

        if(node_split_mat(0,2)==0){
          pred_indices = arma::find(tempvec <= temp_split);
        }else{
          pred_indices = arma::find(tempvec > temp_split);
        }
        //Rcout << "Line 236.\n";

        arma::uvec temp_pred_indices;

        //arma::vec data_subset = testd.col(split);
        arma::vec data_subset = arma_orig_data.col(split);

        data_subset=data_subset.elem(pred_indices);

        //now loop through each row of node_split_mat
        int n=node_split_mat.n_rows;
        //Rcout << "Line 174. i = " << i << ". n = " << n << ".\n";
        //Rcout << "Line 248.\n";

        for(int j=1;j<n;j++){
          int curr_sv=node_split_mat(j,0);
          double split_p = node_split_mat(j,1);

          //data_subset = testd.col(curr_sv-1);
          //Rcout << "Line 255.\n";
          //Rcout << "curr_sv = " << curr_sv << ".\n";
          data_subset = arma_orig_data.col(curr_sv-1);
          //Rcout << "Line 258.\n";

          data_subset=data_subset.elem(pred_indices);

          if(node_split_mat(j,2)==0){
            //split is to the left
            temp_pred_indices=arma::find(data_subset <= split_p);
          }else{
            //split is to the right
            temp_pred_indices=arma::find(data_subset > split_p);
          }
          pred_indices=pred_indices.elem(temp_pred_indices);

          if(pred_indices.size()==0){
            continue;
          }

        }
        //Rcout << "Line 199. i = " << i <<  ".\n";

        //double nodemean=tree_data(terminal_nodes[i]-1,5);
        //IntegerVector predind=as<IntegerVector>(wrap(pred_indices));
        //predictions[predind]= nodemean;
        //term_obs[i]=predind;

        double denom_temp= pred_indices.n_elem+arma::sum(alpha_pars_arma);
        //Rcout << "Line 207. predind = " << predind <<  ".\n";
        //Rcout << "Line 207. denom_temp = " << denom_temp <<  ".\n";
        // << "Line 207. term_node = " << term_node <<  ".\n";

        double num_prod=1;
        double num_sum=0;

        for(int k=0; k<num_cats; k++){
          //assuming categories of y are from 1 to num_cats
          arma::uvec cat_inds= arma::find(orig_y_arma(pred_indices)==k+1);
          double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);

          tree_table1(curr_term-1,5+k)= m_plus_alph/denom_temp ;

          num_prod=num_prod*tgamma(m_plus_alph);
          num_sum=num_sum +m_plus_alph ;
        }


        lik_prod= lik_prod*alph_term*num_prod/tgamma(num_sum);
        //Rcout << "Line 297.\n";


      }
      //Rcout << "Line 301.\n";

    }
    //List ret(1);
    //ret[0] = term_obs;

    //ret[0] = terminal_nodes;
    //ret[1] = term_obs;
    //ret[2] = predictions;
    //return(term_obs);
    //Rcout << "Line 309";

    //return(wrap(arma_tree_table));

    //List ret(2);
    //ret[0]=wrap(arma_tree_table);
    //ret[1]=lik_prod;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////






    //overall_treetables[j]= wrap(tree_table1);


    //double templik = as<double>(treepred_output[1]);

    double templik = pow(lik_prod,beta_par);
    overall_liks(j)= templik;






    //arma::mat arma_tree_table(treetable.begin(), treetable.nrow(), treetable.ncol(), false);
    //arma::mat arma_test_data(testdata.begin(), testdata.nrow(), testdata.ncol(), false);


    //arma::mat arma_tree(tree_data.begin(), tree_data.nrow(), tree_data.ncol(), false);
    //arma::mat testd(test_data.begin(), test_data.nrow(), test_data.ncol(), false);

    //NumericVector internal_nodes=find_internal_nodes_gs(tree_data);

    //NumericVector terminal_nodes=find_term_nodes(treetable);
    //arma::vec arma_terminal_nodes=Rcpp::as<arma::vec>(terminal_nodes);
    //NumericVector tree_predictions;

    //now for each internal node find the observations that belong to the terminal nodes

    //NumericVector predictions(test_data.nrow());

    arma::mat pred_mat(testdata_arma.n_rows,num_cats);
    //arma::vec filled_in(testdata.nrow());


    //List term_obs(terminal_nodes.size());
    if(term_nodes.size()==1){

      //Rcout << "Line 422. \n";


      pred_mat=repmat(tree_table1(0,arma::span(5,5+num_cats-1)),testdata_arma.n_rows,1);


      //Rcout << "Line 424. \n";


      // for(int k=0; k<num_cats; k++){
      // pred_mat(_,k)=rep(treetable(0,5+k),testdata.nrow());
      // }
      //double nodemean=tree_data(terminal_nodes[0]-1,5);				// let nodemean equal tree_data row terminal_nodes[i]^th row , 6th column. The minus 1 is because terminal nodes consists of indices starting at 1, but need indices to start at 0.
      //predictions=rep(nodemean,test_data.nrow());
      //Rcout << "Line 67 .\n";

      //IntegerVector temp_obsvec = seq_len(test_data.nrow())-1;
      //term_obs[0]= temp_obsvec;
      // double denom_temp= orig_y_arma.n_elem+arma::sum(alpha_pars_arma);
      //
      // double num_prod=1;
      // double num_sum=0;

      // for(int k=0; k<num_cats; k++){
      //   //assuming categories of y are from 1 to num_cats
      //   arma::uvec cat_inds= arma::find(orig_y_arma==k+1);
      //   double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
      //   arma_tree_table(0,5+k)= m_plus_alph/denom_temp ;
      //
      //   //for likelihood calculation
      //   num_prod=num_prod*tgamma(m_plus_alph);
      //   num_sum=num_sum +m_plus_alph ;
      // }
      //
      // lik_prod= alph_term*num_prod/tgamma(num_sum);
      //
    }
    else{
      for(unsigned int i=0;i<term_nodes.size();i++){
        //arma::mat subdata=testd;
        int curr_term=term_nodes(i);

        int row_index;
        int term_node=term_nodes(i);


        //WHAT IS THE PURPOSE OF THIS IF-STATEMENT?
        //Why should the ro index be different for a right daughter?
        //Why not just initialize row_index to any number not equal to 1 (e.g. 0)?
        row_index=0;

        // if(curr_term % 2==0){
        //   //term node is left daughter
        //   row_index=terminal_nodes[i];
        // }else{
        //   //term node is right daughter
        //   row_index=terminal_nodes[i]-1;
        // }








        //save the left and right node data into arma uvec

        //CHECK THAT THIS REFERS TO THE CORRECT COLUMNS
        //arma::vec left_nodes=arma_tree.col(0);
        //arma::vec right_nodes=arma_tree.col(1);

        arma::vec left_nodes=tree_table1.col(0);
        arma::vec right_nodes=tree_table1.col(1);



        arma::mat node_split_mat;
        node_split_mat.set_size(0,3);
        //Rcout << "Line 124. i = " << i << " .\n";

        while(row_index!=1){
          //for each terminal node work backwards and see if the parent node was a left or right node
          //append split info to a matrix
          int rd=0;
          arma::uvec parent_node=arma::find(left_nodes == term_node);

          if(parent_node.size()==0){
            parent_node=arma::find(right_nodes == term_node);
            rd=1;
          }

          //want to cout parent node and append to node_split_mat

          node_split_mat.insert_rows(0,1);

          //CHECK THAT COLUMNS OF TREETABLE ARE CORRECT
          //node_split_mat(0,0)=treetable(parent_node[0],2);
          //node_split_mat(0,1)=treetable(parent_node[0],3);

          //node_split_mat(0,0)=arma_tree_table(parent_node[0],3);
          //node_split_mat(0,1)=arma_tree_table(parent_node[0],4);

          node_split_mat(0,0)=tree_table1(parent_node(0),2);
          node_split_mat(0,1)=tree_table1(parent_node(0),3);

          node_split_mat(0,2)=rd;
          row_index=parent_node(0)+1;
          term_node=parent_node(0)+1;
        }

        //once we have the split info, loop through rows and find the subset indexes for that terminal node!
        //then fill in the predicted value for that tree
        //double prediction = tree_data(term_node,5);
        arma::uvec pred_indices;
        int split= node_split_mat(0,0)-1;

        //arma::vec tempvec = testd.col(split);
        arma::vec tempvec = arma_test_data.col(split);


        double temp_split = node_split_mat(0,1);

        if(node_split_mat(0,2)==0){
          pred_indices = arma::find(tempvec <= temp_split);
        }else{
          pred_indices = arma::find(tempvec > temp_split);
        }

        arma::uvec temp_pred_indices;

        //arma::vec data_subset = testd.col(split);
        arma::vec data_subset = arma_test_data.col(split);

        data_subset=data_subset.elem(pred_indices);

        //now loop through each row of node_split_mat
        int n=node_split_mat.n_rows;
        //Rcout << "Line 174. i = " << i << ". n = " << n << ".\n";
        //Rcout << "Line 174. node_split_mat= " << node_split_mat << ". n = " << n << ".\n";


        for(int j=1;j<n;j++){
          int curr_sv=node_split_mat(j,0);
          double split_p = node_split_mat(j,1);

          //data_subset = testd.col(curr_sv-1);
          data_subset = arma_test_data.col(curr_sv-1);

          data_subset=data_subset.elem(pred_indices);

          if(node_split_mat(j,2)==0){
            //split is to the left
            temp_pred_indices=arma::find(data_subset <= split_p);
          }else{
            //split is to the right
            temp_pred_indices=arma::find(data_subset > split_p);
          }
          pred_indices=pred_indices.elem(temp_pred_indices);

          if(pred_indices.size()==0){
            continue;
          }

        }
        //Rcout << "Line 199. i = " << i <<  ".\n";

        //double nodemean=tree_data(terminal_nodes[i]-1,5);
        //IntegerVector predind=as<IntegerVector>(wrap(pred_indices));
        //predictions[predind]= nodemean;
        //term_obs[i]=predind;

        //Rcout << "Line 635. \n";
        //Rcout << "pred_indices = " << pred_indices << ".\n";

        //pred_mat.rows(pred_indices)=arma::repmat(arma_tree_table(curr_term-1,arma::span(5,5+num_cats-1)),pred_indices.n_elem,1);
        pred_mat.each_row(pred_indices)=tree_table1(curr_term-1,arma::span(5,4+num_cats));



        //Rcout << "Line 588. \n";

        // for(int k=0; k<num_cats; k++){
        //   pred_mat(predind,k)=rep(treetable(curr_term-1,5+k),predind.size());
        // }


        // double denom_temp= pred_indices.n_elem+arma::sum(alpha_pars_arma);
        // //Rcout << "Line 207. predind = " << predind <<  ".\n";
        // //Rcout << "Line 207. denom_temp = " << denom_temp <<  ".\n";
        // // << "Line 207. term_node = " << term_node <<  ".\n";
        //
        // double num_prod=1;
        // double num_sum=0;
        //
        // for(int k=0; k<num_cats; k++){
        //   //assuming categories of y are from 1 to num_cats
        //   arma::uvec cat_inds= arma::find(orig_y_arma(pred_indices)==k+1);
        //   double m_plus_alph=cat_inds.n_elem +alpha_pars_arma(k);
        //
        //   arma_tree_table(curr_term-1,5+k)= m_plus_alph/denom_temp ;
        //
        //   num_prod=num_prod*tgamma(m_plus_alph);
        //   num_sum=num_sum +m_plus_alph ;
        // }


        //lik_prod= lik_prod*alph_term*num_prod/tgamma(num_sum);


      }
    }






    //THIS SHOULD BE DIFFERENT IF THE CODE IS TO BE PARALLELIZED
    //EACH THREAD SHOULD OUTPUT ITS OWN MATRIX AND SUM OF LIKELIHOODS
    //THEN ADD THE MATRICES TOGETHER AND DIVIDE BY THE TOTAL SUM OF LIKELIHOODS
    //OR JUST SAVE ALL MATRICES TO ONE LIST


    //pred_mat_overall = pred_mat_overall + templik*pred_mat;
    overall_treetables(j)= pred_mat*templik;

    //overall_treetables(j)= pred_mat;
    overall_liks(j) =templik;

    //arma::mat treeprob_output = get_test_probs(weights, num_cats,
    //                                           testdata,
    //                                           treetable_list[i]  );

    //Rcout << "Line 688. i== " << i << ". \n";

    //double weighttemp = weights[i];
    //Rcout << "Line 691. i== " << i << ". \n";

    //pred_mat_overall = pred_mat_overall + weighttemp*treeprob_output;



  }//end of loop over all trees

}//end of pragma omp code


///////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////


//for(unsigned int i=0; i<overall_treetables.n_elem;i++){
//  pred_mat_overall = pred_mat_overall + overall_liks(i)*overall_treetables(i);
//}


#pragma omp parallel
{
  arma::mat result_private=arma::zeros<arma::mat>(arma_test_data.n_rows,num_cats);
#pragma omp for nowait //fill result_private in parallel
  for(unsigned int i=0; i<overall_treetables.size(); i++) result_private += overall_treetables(i);
#pragma omp critical
  pred_mat_overall += result_private;
}









double sumlik_total= arma::sum(overall_liks);
pred_mat_overall=pred_mat_overall*(1/sumlik_total);
//Rcout << "Line 1141 . \n";
//Rcout << "Line 1146 . \n";

return(wrap(pred_mat_overall));

}
