from scipy import stats
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_model_cv_scores(model, feat_matrix, labels, folds=10, scoring='r2', classification=False):
    
    '''
    I: persisted model object, feature matrix (numpy or pandas datafram), labels, k-folds, scoring metric
    O: mean of scores over each k-fold (float)
    '''
    from sklearn import model_selection, cross_validation
    import numpy as np
    

    if classification:
        skf = cross_validation.StratifiedKFold(labels, n_folds=folds, shuffle=True)

        scores = model_selection.cross_val_score(
            model, 
            feat_matrix, 
            labels, 
            cv=skf, 
            scoring=scoring,
            n_jobs=-1
        )
    
    else:
        scores = model_selection.cross_val_score(
            model, 
            feat_matrix, 
            labels, 
            cv=folds, 
            scoring=scoring, 
            n_jobs=-1
        )

    return np.median(scores), np.mean(scores), scores


def calc_weight_with_regression_rmse(truths, preds):
    #the min caps the feature at 1 and when it's subtracted 
    #the weight is 0 meaning High error features will have a 0 weight
    #low error will have a high weight

    return 1 - min(1, _score_regression_model_rmse(truths, preds))

def _score_regression_model_rmse(truths, preds):
    return ((preds - truths) ** 2).mean() ** .5

def _score_regression_model_r2(truths, preds):
    '''
    just explains fit of the model not whether the model generalizes well
    cv woudl be the best best for that
    '''
    from sklearn.metrics import r2_score

    return r2_score(truths, preds)



def preprocessor_categorical(df_input, target_categorical_col_name):
    from string import punctuation
    df = df_input[:]

    dummy_feature_names_raw_str = ' '.join(set([item for item in df[target_categorical_col_name].values]))
    dummy_feature_names = set(''.join([char for char in dummy_feature_names_raw_str if char not in set(punctuation)]).split())
    
    for name in dummy_feature_names:
        df[name] = df[target_categorical_col_name].apply(lambda x: 1 if name in x else 0)
    
    del df[target_categorical_col_name]
    
    return df

def outlier_scorer(vector_of_points):
    '''
    This returns an absolute score of a data point's outlierness regardless 
    of directionality meaning it's possible to have a largest outlier score 
    and it not be the best deal. 

    ALSO has a shortcoming of finding the true outlierness score and 
    this is designed to address that issue. It measures a point's distance 
    from the regression line not the distance from other points.

    USE-CASE: "deal-score" of a point. Say if we're looking at two independent datasets
    We could make the claim, that one deal is better than other based on this measure.
    '''

    gaps_to_closest_neighbor_of_each_point = [
            find_distance_of_closest_neighbor(i, vector_of_points) \
            for i, distance_of_point in enumerate(vector_of_points)
        ]
    
    return gaps_to_closest_neighbor_of_each_point

def find_distance_of_closest_neighbor(idx_of_subject_point, all_points):
    vector_of_points = all_points[:] 
    pt = vector_of_points[idx_of_subject_point]
    vector_of_points.pop(idx_of_subject_point)
    gaps = [np.abs(pt2 - pt) for pt2 in vector_of_points]
    return min(gaps)

def compute_residuals(model, X, y):
    #truth = 100, pred = 50, residual = -50 | 100-50=50 indicates overprice
    predictions = np.array(model.predict(X))
    residuals = [truth - predictions[idx] for idx, truth in enumerate(y)]
    return residuals

def tranform_to_mse(vector_of_points):
    return [(data_point**2)**0.5 for data_point in vector_of_points]

def get_proba_errors(clf, X_feat_M, y_labels):
    df = pd.DataFrame(clf.predict_proba(X_feat_M), columns=clf.get_classes())
    proba_errors = []
    for idx, row in enumerate(df.values):
        
        row_lookup = dict(list(zip(clf.get_classes(),row)))
        truth_proba = row_lookup[y_labels.iloc[idx]]
        #shouldn't row.sum() be 1.0 100% of the time?
        error = np.abs(row.sum() - truth_proba)
        proba_errors.append(error)

    return proba_errors

def vector_varies_in_values(vector): 
    return len(set(vector)) != 1

def is_binary(vector):
    return len(set(vector)) == 2

def standardize(df_data):        
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_data), columns=df_data.columns)
    return df_scaled, scaler

def get_distance_from_median(vector_of_points):
    return [pt - pd.Series(vector_of_points).median() for pt in vector_of_points]

def transform_to_boxcox_and_normalize(vector_of_points):
    vector_of_points = np.array(vector_of_points)
    #ratio's in the second tuple
    #shift the distribution s.t. that it's positive
    min_point_in_vector_space = np.abs(np.min(vector_of_points))

    box_cox_outliers = stats.boxcox(vector_of_points + (min_point_in_vector_space + 1))[0]
    return StandardScaler().fit_transform(box_cox_outliers)

def dummy_it(input_df, linear_model=False):
    '''
    I: Pandas DataFrame with categorical features
    O: Same df with binarized categorical features
    *check the dummy variable trap thing
    '''
    
    base_case_df = pd.DataFrame()
    categorical_variables = []
    dropped_variables = []
    
    # every column that's not a categorical column, we dummytize
    for col in input_df.columns:
        if str(input_df[col].dtype) != 'object':
            base_case_df = pd.concat([base_case_df, input_df[col]], axis=1)
        else:
            if linear_model:
                dropped_variables.append(pd.get_dummies(input_df[col]).ix[:, -1].name)
                #leaves the last one out
                base_case_df = pd.concat([base_case_df, pd.get_dummies(input_df[col]).ix[:, :-1]], axis=1)
            else:
                base_case_df = pd.concat([base_case_df, pd.get_dummies(input_df[col])], axis=1)
            categorical_variables.append(col)
    

    return base_case_df

def get_names_of_categorical_variables(df):
    return [feature_name for feature_name in df.columns if str(df[feature_name].dtype) == 'object']
