import numpy as np
from helpers import get_model_cv_scores, is_binary


rf_params_regressor = {
    'n_jobs':-1,
    'criterion':'mse', 
    'random_state':4
}

rf_params_clf = {
    'n_jobs':-1, 
    'class_weight':'balanced_subsample', 
    'random_state':4
}


class Regression_and_Classification_Model(object):

    def __init__(self, instatiated_regression_model, instatiated_classification_model, **kwargs):
        self.regressor = instatiated_regression_model
        self.classfier = instatiated_classification_model

    def predict(self, data_points):
        if self.is_target_variable_categorical:
            return self.classfier.predict(data_points)
        else:
            return self.regressor.predict(data_points)
    
    def predict_proba(self, data_points):
        if self.is_target_variable_categorical:
            return self.classfier.predict_proba(data_points)
    
    def fit(self, X, y):
        self.is_target_variable_categorical = (str(y.dtype) == 'object') or is_binary(y)

        if self.is_target_variable_categorical:
            self.classfier.fit(X,y)
        else:
            self.regressor.fit(X,y)

    def cv_score(self, X, y):
        if self.is_target_variable_categorical:
            #massive inefficiency since it calls the folds multiple times
            mean_score = get_model_cv_scores(model=self.classfier, feat_matrix=X, labels=y, scoring='f1_weighted', classification=True)[1]
        else:
            mean_score = get_model_cv_scores(self.regressor, X, y, scoring='r2')[1]
        return mean_score
    
    def get_classes(self):
        if self.is_target_variable_categorical:
            return self.classfier.classes_
        else:
            return None


