from sklearn.base import BaseEstimator, TransformerMixin
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentsStats(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, mails):
        sid = SentimentIntensityAnalyzer()
        sentiment_analysis_result = []
        for mail in mails:
            body_sentences = tokenize.sent_tokenize(mail)
            sentences_stats = map(
                lambda sentence: sid.polarity_scores(sentence), body_sentences)
            stats = {'neg': 0, 'neu': 0, 'pos': 0}
            for sentence_stat in sentences_stats:
                stats['neg'] += sentence_stat['neg']
                stats['neu'] += sentence_stat['neu']
                stats['pos'] += sentence_stat['pos']
            if len(sentences_stats) != 0:
                stats['neg'] /= len(sentences_stats)
                stats['neu'] /= len(sentences_stats)
                stats['pos'] /= len(sentences_stats)
            sentiment_analysis_result.append(stats)
        return sentiment_analysis_result
