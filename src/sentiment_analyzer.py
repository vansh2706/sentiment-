from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from typing import List, Dict, Union


class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('vader_lexicon')

        # Initialize analyzers
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = ' '.join(text.split())

        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]

        return ' '.join(tokens)

    def analyze_textblob(self, text: str) -> str:
        """Analyze sentiment using TextBlob, returns 'positive'|'negative'|'neutral'."""
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text:
            return 'neutral'
        analysis = TextBlob(cleaned_text)
        if analysis.sentiment.polarity > 0.05:
            return 'positive'
        elif analysis.sentiment.polarity < -0.05:
            return 'negative'
        else:
            return 'neutral'

    def analyze_vader(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment using VADER, returns label and compound score."""
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text:
            return {'label': 'neutral', 'score': 0.0}
        scores = self.vader.polarity_scores(cleaned_text)
        comp = scores['compound']
        if comp >= 0.05:
            label = 'positive'
        elif comp <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        return {'label': label, 'score': comp}

    def analyze_batch(self, texts: List[str], method: str = 'vader') -> List[Dict[str, Union[str, float]]]:
        """Analyze sentiment for a batch of texts using specified method (vader|textblob)."""
        results = []
        for text in texts:
            if method == 'textblob':
                sentiment = self.analyze_textblob(text)
            else:
                sentiment = self.analyze_vader(text)
            results.append({'text': text, 'sentiment': sentiment})
        return results