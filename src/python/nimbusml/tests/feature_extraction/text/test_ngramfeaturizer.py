# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import unittest

import numpy as np
import sys
np.set_printoptions(threshold=np.inf)
import six
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.utils import get_X_y
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_equal


class TestNGramFeaturizer(unittest.TestCase):

    def test_ngramfeaturizer(self):
        np.random.seed(0)
        train_file = get_dataset('wiki_detox_train').as_filepath()
        (train,
         label) = get_X_y(train_file,
                          label_column='Sentiment',
                          sep='\t',)
        X_train, X_test, y_train, y_test = train_test_split(
            train['SentimentText'], label)

        # map text reviews to vector space
        texttransform = NGramFeaturizer(
            word_feature_extractor=n_gram(),
            vector_normalizer='None') << 'SentimentText'
        print("X_train Before fit\n")
        print(X_train.values)
        print("Len of X_train before fitting: ", len(X_train))
        X_train = texttransform.fit_transform(X_train[:100])
        print("X_train After fit\n")
        print(X_train)
        print("X_train.iloc[:]\n")
        print(X_train.iloc[:])
        print("X_train.iloc[:].sum()\n")
        print(X_train.iloc[:].sum())
        sum = X_train.iloc[:].sum().sum()
        print("Sum")
        print(sum)
        assert_equal(sum, 30513, "sum of all features is incorrect!")


if __name__ == '__main__':
    unittest.main()
