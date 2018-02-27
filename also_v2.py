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
from helpers import get_model_cv_scores, outlier_scorer, vector_varies_in_values, standardize, get_proba_errors


class attribute_wise_learning_for_outlier_detector(object):
    
    def __init__(self, pandas_dataframe_dataset, model=None):
        
        if not model:

            self.model = Regression_and_Classification_Model(
                    RandomForestRegressor(**rf_params_regressor), 
                    RandomForestClassifier(**rf_params_clf)
                )
        
        self.full_data_set = pandas_dataframe_dataset[:]
            
    def get_recommendations_also(self, name_of_price_field='total_price', top_x=3):
        '''
        *This method shouldn't be designed like this

        It should first filter for all the points under a certain level (budget perhaps) and then run the ALSO aglo.
        instead it runs the algo first and then filters.
        '''
        # self.get_best_deals(name_of_price_field)
        # rank_ordered_full_data_set = self.full_data_set.ix[np.argsort(np.array(self.df_also_residuals.also_residual))]
        rank_ordered_full_data_set = self.full_data_set.ix[np.argsort(np.array(self.df_also_residuals.also_residual))[::-1]]

        #shortcoming, makes an assumption it's decently fit model
        #maybe it should be below the budget. 
        # return rank_ordered_full_data_set[rank_ordered_full_data_set['residuals_by_' + name_of_price_field] < 0.0][:top_x]

        return rank_ordered_full_data_set[:top_x]

    def run_ALSO(self):

        # self.feature_weights, self.df_raw_residuals = self.compute_weights_and_residuals_by_features(self.full_data_set)
        _, self.df_raw_residuals = self.compute_weights_and_residuals_by_features(self.full_data_set)
        self.feature_weights = compute_weights_RRSE(self.df_raw_residuals, self.full_data_set)
        self.df_also_residuals = self.df_raw_residuals.applymap(lambda x: x**2)
        self.df_also_residuals, _ = standardize(self.df_also_residuals)
        self.df_also_residuals = self.apply_feature_weights(self.df_also_residuals)
        
        #adding some measure of dispersion should give us some inkling 
        #of volatility of a point 
        #std is fine for now but it doesn't properly encapsulate change 
        #of sign, which is quite relavent
        self.df_also_residuals['also_std'] = self.df_raw_residuals.std(axis=1)        
        self.df_also_residuals['also_residual'] = self.df_also_residuals.sum(axis=1)
        self.df_also_residuals['also_residual'] = self.df_also_residuals['also_residual'].apply(lambda x: x**0.5)
        self.df_also_residuals['also_outlier_score'] = outlier_scorer(self.df_also_residuals['also_residual'])
                
        return self.df_also_residuals
    

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
            
            #r2 scores everything on this version, look up v1 to see how categorical scoring
            #is implemented
            _, residuals, weight = compute_cv_scores_and_residuals(X, y)
            
            #it's important not to transform the residuals at this stage because 
            #for the special-case scenario, the sign is important
            residuals_by_feature[target_variable_name] = residuals
            feature_relevence_weights_dict[target_variable_name] = weight if weight > 0.0 else 0.0
    
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


    # def compute_residuals_for_target_variable(self, training_data, target_variable_name):
    #     X = dummy_it(training_data.drop(target_variable_name, axis=1))
    #     y = training_data[target_variable_name]

    #     self.model.fit(X, y)
        
    #     return compute_residuals(self.model, X, y)

    # def cv_score(self, target_variable_name):
        
    #     try:
    #         if self.feature_weights[target_variable_name]: 
    #             return self.feature_weights[target_variable_name]
    #     except:        
    #         X = dummy_it(self.full_data_set.drop(target_variable_name, axis=1))
    #         y = self.full_data_set[target_variable_name]
            
            
    #         return get_model_cv_scores(self.model.fit(X, y), X, y, scoring='r2')[1]


    def compute_residuals_for_target_variable(self, training_data, target_variable_name):
        X = dummy_it(training_data.drop(target_variable_name, axis=1))
        y = training_data[target_variable_name]
        _, residuals, _ = compute_cv_scores_and_residuals(X, y)
        return residuals
        
    

    def compute_budget(self, name_of_price_field='total_price', func=np.median):
        
        X = dummy_it(self.full_data_set.drop(name_of_price_field, axis=1))
        y = self.full_data_set[name_of_price_field]
        
        self.model.fit(X, y)
        
        predictions = np.array(self.model.predict(X))
    
        return func(predictions)


rf_params_regressor = {
    'n_jobs':-1,
    'criterion':'mse', 
    'random_state':4
}



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def compute_cv_scores_and_residuals(X, y):
    kf = KFold(n_splits=10)

    r2_scores=[]
    raw_residuals=[]
    corresponding_indices=[]

    for train_idx, test_idx in kf.split(X):

        X_train = X.ix[train_idx]
        y_train = y.ix[train_idx]
        
        X_test = X.ix[test_idx]
        y_test = y.ix[test_idx]

        rf = RandomForestRegressor(**rf_params_regressor)
        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)
        r2_scores.append(r2_score(y_test, predictions))
        
        #truth = 100, pred = 50, residual = -50 | 100-50=50 indicates overprice
        raw_residuals.extend([truth - predictions[idx] for idx, truth in enumerate(y_test)])
        corresponding_indices.extend(test_idx)

        
    sorted_list = sorted(zip(corresponding_indices, raw_residuals), key=lambda x:x[0])
    idx, raw_residuals_sorted = list(zip(*sorted_list))
    
    return idx, raw_residuals_sorted, np.mean(r2_scores)

def compute_weights_RRSE(df_residuals, df_feat_M):
    # https://link.springer.com/content/pdf/10.1007%2Fs10994-015-5507-y.pdf
    df = df_residuals.applymap(lambda x: x**2)
    
    weights={}
    for feat_name in df.columns: 
        R_feat = np.sqrt( \
            df[feat_name].sum()/
            sum([(truth - df_feat_M[feat_name].mean())**2 for truth in df_feat_M[feat_name]])
        )
        
        weights[feat_name] = 1 - min(1,R_feat)
    
    return weights

def plot_optimal_number_of_components_variance(scale_feature_collinear_feature_M, variance_threshold=0.99):
    '''
    typical rule of thumb says keep the number of compenonents that get ~90% of 
    variance (50) by looking at the cumsum graph. but if we look at the scree plot 
    and abide by kairsers rule which argues to keep as many explanatory components 
    i.e. when the slope doesn't change -- elbow method
    '''
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    n_col = scale_feature_collinear_feature_M.shape[1]
    pca = PCA(n_components=n_col)
    train_components = pca.fit_transform(scale_feature_collinear_feature_M)

    pca_range = np.arange(n_col) + 1
    xbar_names = ['PCA_%s' % xtick for xtick in pca_range]
    cumsum_components_eig = np.cumsum(pca.explained_variance_ratio_)

    target_component = np.where(cumsum_components_eig > variance_threshold)[0][0] + 1

    print ('number of components that explain target amt of variance explained: ' \
            + str(target_component) + ' @ ' + str(cumsum_components_eig[target_component - 1]))

    kaiser_rule = len(pca.explained_variance_ratio_[np.mean(pca.explained_variance_ratio_) > pca.explained_variance_ratio_])
    label1_str = str(100 * cumsum_components_eig[target_component - 1])[:3] + '%'

    #cumsum plot                                                 
    plt.axvline(target_component, color='r', ls='--', alpha=.3, label= str(100 * cumsum_components_eig[target_component - 1])[:4] + '%')
    plt.axvline(kaiser_rule, ls='--', alpha=.3, label= str(100 * cumsum_components_eig[kaiser_rule])[:4] + '%')
    plt.plot(cumsum_components_eig)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.legend(loc='best')
    plt.show()
    
    #Scree
    plt.axvline(target_component, color='r', ls='--', alpha=.3, label= str(100 * cumsum_components_eig[target_component - 1])[:4] + '%')
    plt.axvline(kaiser_rule, ls='--', alpha=.3, label= str(100 * cumsum_components_eig[kaiser_rule])[:4] + '%')
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('variance of component (eigenvalues)')
    plt.legend(loc='best')
    plt.show()
    
    return target_component
