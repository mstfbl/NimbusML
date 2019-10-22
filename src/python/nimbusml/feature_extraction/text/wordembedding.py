# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
WordEmbedding
"""

__all__ = ["WordEmbedding"]


from sklearn.base import TransformerMixin

from ...base_transform import BaseTransform
from ...internal.core.feature_extraction.text.wordembedding import \
    WordEmbedding as core
from ...internal.utils.utils import trace


class WordEmbedding(core, BaseTransform, TransformerMixin):
    """

    Word Embeddings transform is a text featurizer which converts vectors
    of text tokens into sentence vectors using a pre-trained model.

    .. remarks::
        WordEmbeddings wrap different embedding models, such as
        Sentiment Specific Word Embedding(SSWE). Users can specify
        which embedding to use. The
        available options are various versions of `GloVe Models
        <https://nlp.stanford.edu/projects/glove/>`_, `FastText
        <https://en.wikipedia.org/wiki/FastText>`_, and `Sswe
        <https://anthology.aclweb.org/P/P14/P14-1146.pdf>`_.


    :param columns: a dictionary of key-value pairs, where key is the output
        column name and value is the input column name.

        * Only one key-value pair is allowed.
        * Input column type:
         `Vector Type </nimbusml/concepts/types#vectortype-column>`_.
        * Output column type:
         `Vector Type </nimbusml/concepts/types#vectortype-column>`_.
        * If the output column name is same as the input column name, then
        simply specify ``columns`` as a string.

        The << operator can be used to set this value (see
        `Column Operator </nimbusml/concepts/columns>`_)

        For example
         * WordEmbedding(columns={'out1':'input1',)
         * WordEmbedding() << {'ou1': 'input1'}

        For more details see `Columns </nimbusml/concepts/columns>`_.

    :param model_kind: Pre-trained model used to create the vocabulary.
        Available options are: 'GloVe50D', 'GloVe100D', 'GloVe200D',
        'GloVe300D', 'GloVeTwitter25D', 'GloVeTwitter50D',
        'GloVeTwitter100D', 'GloVeTwitter200D', 'FastTextWikipedia300D',
        'SentimentSpecificWordEmbedding'.

    :param custom_lookup_table: Filename for custom word embedding model.

    :param params: Additional arguments sent to compute engine.

    .. note::

        As ``WordEmbedding`` requires a column with text vector, e.g.
        <'This', 'is', 'good'>, users need to create an input column by:

        * concatenating columns with TX type,
        * or using the ``output_tokens_column_name`` for ``NGramFeaturizer()`` to
        convert a column with sentences like "This is good" into <'This',
        'is', 'good'>.


        In the following example, after the ``NGramFeaturizer``, features
        named *ngram.__* are generated.
        A new column named *ngram_TransformedText* is also created with the
        text vector, similar as running ``.split(' ')``.
        However, due to the variable length of this column it cannot be
        properly converted to pandas dataframe,
        thus any pipelines/transforms output this text vector column will
        throw errors. However, we use *ngram_TransformedText* as the input to
        ``WordEmbedding``, the
        *ngram_TransformedText* column will be overwritten by the output from
        ``WordEmbedding``. The output from ``WordEmbedding`` is named
        *ngram_TransformedText.__*

    .. seealso::
        :py:class:`NGramFeaturizer
        <nimbusml.feature_extraction.text.NGramFeaturizer>`,
        :py:class:`Sentiment
        <nimbusml.feature_extraction.text.Sentiment>`.

    .. index:: dnn, features, embedding

    Example:
       .. literalinclude:: /../nimbusml/examples/WordEmbedding.py
              :language: python
    """

    @trace
    def __init__(
            self,
            model_kind='SentimentSpecificWordEmbedding',
            custom_lookup_table=None,
            columns=None,
            **params):

        if columns:
            params['columns'] = columns
        BaseTransform.__init__(self, **params)
        core.__init__(
            self,
            model_kind=model_kind,
            custom_lookup_table=custom_lookup_table,
            **params)
        self._columns = columns

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
