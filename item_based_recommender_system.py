'''
comprehensive notes found here: prototype_item-based_recommender_model.ipynb

i) finds the ideal example 
ii) ranks items closest to this ideal example

todos:
- incorporate the rest of the features
- try out different distance functions:
    manhattan,
    gower (https://github.com/scikit-learn/scikit-learn/issues/5884),
    euclidean,
    mahalnobas
- implement the above and use t-SNE as the inpection tool for contrast
- should we find datasets (similar ones ideally to judge the recommender systems)
- for the square distance metric: 
    https://stackoverflow.com/questions/39203662/euclidean-distance-matrix-using-pandas

Where do recommendations from the kayak bit fall, start with the first 5?

having the weights like so isnt appropriate as it leaves out all of the other features completely
WTS = {
    'price': 4.0,
    'leg_1_stops' : 4.0,
    'leg_2_stops' : 4.0,
}

a better way to use it would be like so...
WTS = {
    'price': 4.0,
    'leg_1_duration' : 1.0,
    'leg_2_duration' : 1.0,
    'leg_1_stops' : 4.0,
    'leg_2_stops' : 4.0,
    'leg_1_depart_time_red_eye_hours' : 1.0,
    'leg_2_depart_time_red_eye_hours' : 1.0
}

'''

import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from helpers import standardize
import numpy as np

class Item_Based_Recommender_System(object):

    def __init__(self, pandas_dataframe_dataset, wts, target_item=None):
        self.full_data_set = pandas_dataframe_dataset[:]
        self.full_data_set_scaled, self.scaler = standardize(self.full_data_set)

        self.wts = wts
        self.features_of_interest = list(self.wts.keys())
        
        if not target_item:
            self.target_item = self.find_ideal_item(self.full_data_set_scaled)
        else:            
            self.target_item = self.scaler.transform([self.target_item[col] for col in self.features_of_interest])

        
    def find_ideal_item(self, df_scaled_data_set):
        '''
        just finds all the ideal amounts of each attribute, it's 0 in this case.
        '''
        unpacked_array = self.scaler.transform(np.zeros(len(self.features_of_interest)).reshape(-1, 1))[0]
        
        return dict(zip(self.features_of_interest, unpacked_array))
        

    def get_recommendations(self, top_x=5):
        #note: for a similarity/distance based approach we'd want to sort is accordingly. 

        #distance: closest transalates to smallest
        #similarity: closest transalates to largest
        return self.full_data_set.ix[np.argsort(self.compute_wtd_distances(self.wts))][:top_x]

    def compute_wtd_distances(self, wts, distance_func=euclidean_distances):
        df_relevent_subset = self.full_data_set_scaled[self.features_of_interest]        
        target_item_wtd, df_relevent_subset_wtd = self.apply_weights(self.target_item, df_relevent_subset)        
        distances = []

        for idx, row in self.full_data_set_scaled.iterrows():
            distances.append(
                    distance_func(
                        np.array(list(target_item_wtd.values())).reshape(1, -1), 
                        row[list(target_item_wtd.keys())].reshape(1, -1)
            )[0][0])
        '''
        #time profile modification * will also get rid of the warnings
        distances = df_relevent_subset_wtd[:1].apply(
            lambda row: 
                some_distance_func(
                    list(target_item_wtd.values()), 
                    row[list(target_item_wtd.keys())]
                ),
            axis=0
        )
        '''
        return distances

    def apply_weights(self, vector_dict, df):

        df_wtd = df[:]

        for feature_name, wt in self.wts.items():

            vector_dict[feature_name] = vector_dict[feature_name] * wt
            df_wtd[feature_name] = df_wtd[feature_name].apply(lambda x: x * wt)

        return vector_dict, df_wtd



if __name__ == '__main__':

    df = pd.read_csv('kayak_experiment6_matt.csv', index_col=[0])

    WTS = {
        'price': 4.0,
        'leg_1_duration' : 1.0,
        'leg_2_duration' : 1.0,
        'leg_1_stops' : 4.0,
        'leg_2_stops' : 4.0,
        'leg_1_depart_time_red_eye_hours' : 1.0,
        'leg_2_depart_time_red_eye_hours' : 1.0
    }


    nn_system = Item_Based_Recommender_System(df.drop(['airline','text'], axis=1), WTS)
    print (nn_system.get_recommendations())