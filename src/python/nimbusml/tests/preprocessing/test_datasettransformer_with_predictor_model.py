# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier, OnlineGradientDescentRegressor
from nimbusml.preprocessing import DatasetTransformer
from nimbusml.preprocessing.filter import RangeFilter
from nimbusml import FileDataStream 

seed = 0

train_data = {'c0': ['a', 'b', 'a', 'b'],
              'c1': [1, 2, 3, 4],
              'c2': [2, 3, 4, 5]}
train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                            'c2': np.float64})

test_data = {'c0': ['a', 'b', 'b'],
             'c1': [1.5, 2.3, 3.7],
             'c2': [2.2, 4.9, 2.7]}
test_df = pd.DataFrame(test_data).astype({'c1': np.float64,
                                          'c2': np.float64})


class TestDatasetTransformerWithPredictorModel(unittest.TestCase):

    def test_dataset_transformer_with_predictor_model(self):
        train_df_updated = train_df.drop(['c0'], axis=1)
        test_df_updated = test_df.drop(['c0'], axis=1)
        rf_max = 4.5
        # Create combined pipeline
        print("---------------WORKING PIPELINE---------------")
        transform_pipeline = Pipeline([RangeFilter(min=0.0, max=rf_max) << 'c2'])
        print("@@@@@@@@LINE 42 - Before calling transform_pipeline.fit")
        transform_pipeline.fit(train_df_updated)
        print("@@@@@@@@LINE 44")
        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline.model),
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        ], random_state=seed)
        print("@@@@@@@@LINE 49 - Before calling combined_pipeline.fit")
        combined_pipeline.fit(train_df_updated)
        print("---------------Transform results---------------")
        resultTransform = combined_pipeline.transform(test_df_updated)
        print("@@@@@@@@LINE 53")
        print(resultTransform)

        print("---------------Predict results---------------")
        resultPredict = combined_pipeline.predict(test_df_updated)
        print("@@@@@@@@LINE 58")
        print(resultPredict)

        print("---------------NON-WORKING PIPELINE 1---------------")
        transform_pipeline2 = Pipeline([RangeFilter(min=0.0, max=rf_max) << 'c2', OnlineGradientDescentRegressor(label='c2', feature=['c1'])], random_state=seed)
        print("@@@@@@@@LINE 63 - Before calling transform_pipeline2.fit")
        transform_pipeline2.fit(train_df_updated)
        print("@@@@@@@@LINE 65")
        print("---------------NON-WORKING PIPELINE 2---------------")
        combined_pipeline2 = Pipeline([DatasetTransformer(transform_model=transform_pipeline2.model)])
        print("@@@@@@@@LINE 68")

        print("---------------Broken pipeline results---------------")
        print("@@@@@@@@LINE 71 - Before calling combined_pipeline2.fit")
        combined_pipeline2.fit(train_df_updated)
        print("@@@@@@@@LINE 73 - Before calling combined_pipeline2.transform")
        resultTransform2 = combined_pipeline2.transform(test_df_updated)
        print("@@@@@@@@LINE 75")
        print(resultTransform2)

if __name__ == '__main__':
    unittest.main()

