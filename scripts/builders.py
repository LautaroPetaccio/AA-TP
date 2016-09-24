from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import simple_features as sf


def build_simple_extractors():
    extractors = [('subject_length', sf.subject_length),
                  ('subject_spaces',
                   sf.subject_spaces),
                  ('subject_caps',
                   sf.subject_caps),
                  ('body_length',
                   sf.body_length),
                  ('body_spaces',
                   sf.body_spaces),
                  ('body_caps',
                   sf.body_caps),
                  ('body_sentences',
                   sf.body_sentences),
                  ('has_html',
                   sf.has_html),
                  ('has_image',
                   sf.has_image)
                  ]
    return 'simple_features', sf.SimpleFeaturesExtractor(extractors)


def build_vectorizer_extractor(**kwargs):
    return ('tfidf', TfidfVectorizer(stop_words='english', strip_accents='ascii', sublinear_tf=True, **kwargs))


def build_dimensionality_reductor(reductor_type, **kwargs):
    if reductor_type is None:
        return None
    elif reductor_type == 'k_best':
        return 'k_best', SelectKBest(chi2, **kwargs)
    elif reductor_type == 'l1':
        return 'l1', SelectFromModel(**kwargs)
    elif reductor_type == 'pca':
        return 'pca', TruncatedSVD(**kwargs)


def build_column_pipeline(column, pipeline_type, tf_idf_ngram_range,
                          tf_idf_min_df, tf_idf_max_df, reduced_df_ngram_range,
                          reduced_min_df, reduced_max_df, k_best_k, l1_C, pca_n_components):
    pipeline = [('selector', sf.ColumnSelectorExtractor(column))]
    if pipeline_type == 'tfidf':
        pipeline.append(build_vectorizer_extractor(
            ngram_range=tf_idf_ngram_range, min_df=tf_idf_min_df, max_df=tf_idf_max_df), )
    else:
        pipeline.append(build_vectorizer_extractor(
            ngram_range=reduced_df_ngram_range, min_df=reduced_min_df, max_df=reduced_max_df))

        if pipeline_type == 'k_best':
            pipeline.append(build_dimensionality_reductor(
                pipeline_type, k=k_best_k))
        elif pipeline_type == 'l1':
            pipeline.append(build_dimensionality_reductor(
                pipeline_type, estimator=LinearSVC(penalty="l1", dual=False, C=l1_C)))
        elif pipeline_type == 'pca':
            pipeline.append(build_dimensionality_reductor(
                pipeline_type, n_components=pca_n_components))
        else:
            return None
    return column, Pipeline(pipeline)


def build_subject_pipeline(pipeline_type):
    return build_column_pipeline('subject', pipeline_type, (1, 1), 0.001, 0.7, (1, 1), 0.0001, 0.95, 20, 1.0, 20)


def build_body_pipeline(pipeline_type):
    return build_column_pipeline('body', pipeline_type, (1, 1), 0.001, 0.7, (1, 1), 0.0001, 0.95, 50, 5.0, 50)


def build_classifier(classifier_type, **kwargs):
    classifier_dict = {'dt': DecisionTreeClassifier(**kwargs),
                       'random_forest': RandomForestClassifier(**kwargs),
                       'bernoulli_nb': BernoulliNB(**kwargs),
                       'multinomial_nb': MultinomialNB(**kwargs),
                       'knn': KNeighborsClassifier(**kwargs),
                       'svm': SVC(**kwargs)}

    if not classifier_type in classifier_dict:
        return None

    return classifier_type, classifier_dict[classifier_type]


def build_pipeline(simple_features=True, subject_pipeline_type=None, body_pipeline_type=None, classifier_type=None):
    extractors = []
    names = []

    if simple_features:
        # Extractor de atributos simples
        extractors.append(build_simple_extractors())
        names.append('simple_features')

    if subject_pipeline_type is not None:
        # Pipeline para extraer atributos de vectorizacion del asunto del mail
        extractors.append(build_subject_pipeline(subject_pipeline_type))
        names.append('subject_%s' % subject_pipeline_type)

    if body_pipeline_type is not None:
        # Pipeline para extraer atributos de vectorizacion del cuerpo del mail
        extractors.append(build_body_pipeline(body_pipeline_type))
        names.append('body_%s' % body_pipeline_type)

    extractors_count = len(extractors)
    if extractors_count == 0:
        return None

    # PCA devuelve una matriz con valores negativos, cosa que ni BernoulliNB
    # ni MultinomialNB soportan
    if classifier_type in ['bernoulli_nb', 'multinomial_nb'] and 'pca' in [subject_pipeline_type, body_pipeline_type]:
        return None

    if classifier_type is not None:
        # Usamos FeatureUnion para combinar los distintos extractores de
        # atributos
        names.append(classifier_type)
        return '__'.join(names), Pipeline([('feature_extractors', FeatureUnion(extractors)), build_classifier(classifier_type)])
    else:
        return None


def build_all_pipelines(subject_pipeline_types, body_pipeline_types, classifier_types):
    pipelines = []
    for simple_features in [True, False]:
        for subject_pipeline_type in subject_pipeline_types:
            for body_pipeline_type in body_pipeline_types:
                for classifier_type in classifier_types:
                    pipeline_tuple = build_pipeline(simple_features=simple_features,
                                                    subject_pipeline_type=subject_pipeline_type,
                                                    body_pipeline_type=body_pipeline_type,
                                                    classifier_type=classifier_type)
                    if pipeline_tuple is not None:
                        pipelines.append(pipeline_tuple)

    return pipelines
