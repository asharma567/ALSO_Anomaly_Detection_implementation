'''
todos
-----
### FIXES
- score the points during the CV process i.e. don't train on the full set and then score.
- score the point via mse instead of abs, mse. It'll seperate the outliers better.
- figure out how to properly do categoricals.

- create tutorial on how to interpret an outlier.
- create a better preferential wieght learner
- RECs3: train a model on a bunch of budgets


ALSO V3

V3: Adding PCA to address the multicollinearity.

This is more applicable to the use-case of anomaly detection as opposed to flights. Reason being, it adds greatly to the time-complexity. 

Say for example, there's 2 features which are copies of one another it'll have double the impact at the stage of findig the outlier score. So to address this issue there will be PCA step that takes place before everything else.

The downfall, the components are not interpretible. Hence, the "learned weights" should have a natural decay with increasing number of components i.e. the higher the number of components the less significant. 


send to charu
    ask him about the roc_auc
    #* the scoring functions here needs to be crossvalidated
    #the major differenc I could see here rmse is much more aggressive than
    #r2 that is it'll more likely call something a 0 weight because of the
    #threshold of 1

    #Charu: what's the motivation behind using the RMSE instead of R^2 for coming up with relevance weights

    #to include the relevance weight of categgorical variables I've used the f1_score. I know this is threshold sensitive, 
    the other option is to use the roc_auc score and transform it to a 0 to 1.0 scale

    #the way I've computed the residuals is by summing each example across all target_variable_errors.
    #for computing the residuals for a continuous target variable it's trivial but for categorical, it wasnt'.

    #I took the delta of the probability of truth class and sum of all the probabilities of the other classes. 
    #I called this the error.

#LOO is another improvement to the residual computer
#a method could also be built to find another type of outlier which is the most disagreeable examples
#just had a idea for a v2 to create a competign flight algo which is trained on all the flight queries
#needs to be removed from the learned-weights 'residuals_by_total_price':

- kayak scraper for recommendations benchmarking DONE
- experimental apparatus
    - time profile DONE
    - sendout results to budgeting team DONE
    - tuning and gridsearching
    to aid in the time profile one could not do the CV or 
    reduce the many reduncies in the algo
- make it generalizable across any dataset DONE
- check is categorical is working DONE
- a write up on the algo and how it works DONE
#find the intersection between the basic dependency outlier model DONE
refactor for readability DONE
get the git repo to work DONE
get cv to work DONE
fix bugs DONE
'''

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from model_class import Regression_and_Classification_Model, rf_params_regressor, rf_params_clf
from helpers import dummy_it, preprocessor_categorical, compute_residuals, transform_to_boxcox_and_normalize
from helpers import outlier_scorer, vector_varies_in_values, standardize, get_proba_errors


class attribute_wise_learning_for_outlier_detector(object):
    
    def __init__(self, pandas_dataframe_dataset, model=None):
        
        if not model:

            self.model = Regression_and_Classification_Model(
                    RandomForestRegressor(**rf_params_regressor), 
                    RandomForestClassifier(**rf_params_clf)
                )
        
        self.full_data_set = pandas_dataframe_dataset[:]
            
    def get_recommendations_also(self, name_of_price_field='total_price', top_x=3):
        self.get_best_deals(name_of_price_field)
        #should this be sorted after transformation abs residuals?
        rank_ordered_full_data_set = self.full_data_set.ix[np.argsort(np.array(self.df_residuals.also_residual))]

        return rank_ordered_full_data_set[rank_ordered_full_data_set['residuals_by_' + name_of_price_field] < 0.0][:top_x]

    def outlier_score_all_datapoints(self):

        self.feature_weights, self.df_residuals = self.compute_weights_and_residuals_by_features(self.full_data_set)
        df_residuals_scaled, _ = standardize(self.df_residuals)
        df_residuals_scaled_wtd = self.apply_feature_weights(df_residuals_scaled)
        
        return self.compute_outlier_scores(df_residuals_scaled_wtd)
    
    def compute_outlier_scores(self, df_residuals_scaled_wtd):
        #should we be taking taking a straight sum across here?
        #add some measure of volatility i.e. std
        self.df_residuals['also_residual'] = df_residuals_scaled_wtd.sum(axis=1)        
        self.df_residuals['also_abs_residual'] = self.df_residuals['also_residual'].apply(np.abs)
        self.df_residuals['also_outlier_score'] = outlier_scorer(self.df_residuals['also_abs_residual'])
        
        return self.df_residuals

    def apply_feature_weights(self, df_residuals_scaled): 
        df_residuals_scaled_wtd = df_residuals_scaled[:]        
        for col in df_residuals_scaled.columns:
            df_residuals_scaled_wtd[col] = df_residuals_scaled[col] * self.feature_weights[col]
        return df_residuals_scaled_wtd

    def compute_weights_and_residuals_by_features(self, training_data):
        feature_relevence_weights_dict = {}
        residuals_by_feature = {}

        for target_variable_name in training_data.columns:

            if not vector_varies_in_values(training_data[target_variable_name]): 
                feature_relevence_weights_dict[target_variable_name] = 0.0
                continue
            
            X, y = self._get_XY_and_add_dummies(training_data, target_variable_name)
            self.model.fit(X, y)
            
            #this is optimized for r2 so if the weight was negative meaning 
            #it fits the data poorly
            weight = self.model.cv_score(X, y) if self.model.cv_score(X, y) > 0 else 0
            feature_relevence_weights_dict[target_variable_name] = weight
            
            if self.model.is_target_variable_categorical:                
                residuals_by_feature[target_variable_name] = get_proba_errors(self.model, X, y)
            else:
                residuals_by_feature[target_variable_name] = compute_residuals(self.model, X, y)
    
        return feature_relevence_weights_dict, pd.DataFrame(residuals_by_feature)

    def _get_XY_and_add_dummies(self, training_data, target_variable_name):
        preprocessed_data = training_data[:]
        
        y = preprocessed_data.pop(target_variable_name)
        X = dummy_it(preprocessed_data)
        
        return X, y

    def get_recommendations_target_variable(self, name_of_price_field='total_price', top_x=3, stds=None):

        best_to_average_deal_sorted_dataset, _ = self.get_best_deals(name_of_price_field)
        
        if not stds:
            return best_to_average_deal_sorted_dataset[:top_x]
        else:
            return best_to_average_deal_sorted_dataset[best_to_average_deal_sorted_dataset.boxcoxed_residuals_by_total_price < stds]

    def get_best_deals(self, name_of_price_field):
        '''
        best_to_worst_deal_sorted_dataset is included for academic purposes
        best_to_average_deal_sorted_dataset is within the decision set we care most about
        '''

        residuals = self.compute_residuals_for_target_variable(self.full_data_set, name_of_price_field)
        
        self.full_data_set['residuals_by_' + name_of_price_field] = np.array(residuals).reshape(-1,1)
        indices_best_to_worst_deal = np.argsort(residuals)

        best_to_worst_deal_sorted_dataset = self.full_data_set.ix[indices_best_to_worst_deal]
        best_to_worst_deal_sorted_dataset['boxcoxed_residuals_by_' + name_of_price_field] = transform_to_boxcox_and_normalize(best_to_worst_deal_sorted_dataset['residuals_by_' + name_of_price_field])
        best_to_worst_deal_sorted_dataset['skew'] = skew(best_to_worst_deal_sorted_dataset['boxcoxed_residuals_by_' + name_of_price_field])
        best_to_worst_deal_sorted_dataset['kurtosis'] = kurtosis(best_to_worst_deal_sorted_dataset['boxcoxed_residuals_by_' + name_of_price_field])
        
        best_to_average_deal_sorted_dataset = best_to_worst_deal_sorted_dataset[best_to_worst_deal_sorted_dataset['residuals_by_' + name_of_price_field] < 0.0]
        best_to_average_deal_sorted_dataset['outlier_score'] = outlier_scorer(list(best_to_average_deal_sorted_dataset['residuals_by_' + name_of_price_field]))

        best_to_average_deal_sorted_dataset['boxcoxed_residuals_by_' + name_of_price_field] = transform_to_boxcox_and_normalize(best_to_average_deal_sorted_dataset['residuals_by_' + name_of_price_field])
        best_to_average_deal_sorted_dataset['skew'] = skew(best_to_average_deal_sorted_dataset['boxcoxed_residuals_by_' + name_of_price_field])
        best_to_average_deal_sorted_dataset['kurtosis'] = kurtosis(best_to_average_deal_sorted_dataset['boxcoxed_residuals_by_' + name_of_price_field])
        
        return best_to_average_deal_sorted_dataset, best_to_worst_deal_sorted_dataset


    def compute_residuals_for_target_variable(self, training_data, target_variable_name):
        X = dummy_it(training_data.drop(target_variable_name, axis=1))
        y = training_data[target_variable_name]

        self.model.fit(X, y)
        
        return compute_residuals(self.model, X, y)
    


    def compute_budget(self, name_of_price_field='total_price', func=np.median):
        
        X = dummy_it(self.full_data_set.drop(name_of_price_field, axis=1))
        y = self.full_data_set[name_of_price_field]
        
        self.model.fit(X, y)
        
        predictions = np.array(self.model.predict(X))
    
        return func(predictions)

