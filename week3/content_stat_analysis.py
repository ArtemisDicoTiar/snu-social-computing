import json
import sys
from collections import Counter
from datetime import datetime
from functools import reduce
from pprint import pprint
from typing import List

import nltk
from konlpy.tag import Okt
from nltk import PorterStemmer

from pathlib import Path

# --- The directories --- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


# For Apple Silicon
JVM_PATH = '/opt/homebrew/opt/openjdk@17/bin/java'


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

from nltk.corpus import stopwords
twt_stop_words = ['.', ',', '--', '\'s', '?', '!', ')', '(', ':', '\'', '\'re', '"', '-', '}', '{', u'—', 'rt', 'http', 't', 'co', '@', '#',]
en_stop_words = stopwords.words('english') + twt_stop_words
en_stemmers = PorterStemmer()

ko_tokenizer = Okt(jvmpath=JVM_PATH)

class ContentStat:
    def __init__(self, data_name: str, tokenizer_lang: str = "ko"):
        self.data_name = data_name
        self.file_name = f"{data_name}.json"

        self.twitter_data = self._load_twitter_json()

        self.tokenizer_lang = tokenizer_lang

        self._base_tweet_created()
        self._base_user_created()
        self._base_num_tweets_by_user()
        self._base_num_followers()
        self._base_hashtags()
        self._base_tweet_lang()
        self._base_content_words()

    # ============= Data Load ============= #
    def _load_twitter_json(self):
        with (DATA_DIR / self.file_name).open() as f:
            statuses = json.loads(f.read())
        return statuses

    # ============= Utils ============= #
    @staticmethod
    def _lower_strip_split(sentence: str) -> List[str]:
        """
        @param sentence: raw string sentence.
        @return: lowered, stripped sentence split by single space.

        """
        return sentence.lower().strip().split()

    @staticmethod
    def _remove_stopwords(terms: List[str]) -> List[str]:
        """
        @param terms: terms already lowered, striped, split
        @return: list of terms without stopwords
        """
        return list(filter(lambda x: x not in STOPWORDS, terms))

    @staticmethod
    def _remove_punctuation(terms: List[str]) -> List[str]:
        """
        @param terms: terms already lowered, striped, split
        @return:
        """
        punc_removed = [word.strip(punctuation) for word in terms]
        return [word for word in punc_removed if word]

    def _regularize_string(self, sentence: str) -> str:
        """
        @param sentence: raw sentence
        @return: regularized sentence
        """
        return " ".join(
            self._remove_stopwords(
                self._remove_punctuation(
                    self._lower_strip_split(sentence)
                )
            )
        )

    # ============= Tweet Created ============= #
    def _base_tweet_created(self) -> None:
        self.tweet_created = [
            datetime.strptime(status["created_at"], "%a %b %d %H:%M:%S %z %Y")
            for status in self.twitter_data
        ]

        self.tweet_crated_as_unix_times = [
            dt.timestamp()
            for dt in self.tweet_created
        ]

    # ============= User Created ============= #
    def _base_user_created(self):
        self.user_created = [
            datetime.strptime(status["user"]["created_at"], "%a %b %d %H:%M:%S %z %Y")
            for status in self.twitter_data
        ]

        self.user_crated_as_unix_times = [
            dt.timestamp()
            for dt in self.user_created
        ]

    # ============= Tweets <- User ============= #
    def _base_num_tweets_by_user(self):
        self.num_tweets_by_user = [
            status["user"]["statuses_count"]
            for status in self.twitter_data
        ]

    # ============= Followers ============= #
    def _base_num_followers(self):
        self.num_followers = [
            status["user"]["followers_count"]
            for status in self.twitter_data
        ]

    # ============= Hashtags ============= #
    def _base_hashtags(self):
        self.hashtags_cnt = Counter([
            record['text']
            for record in
            list(reduce(
                lambda i, j: i + j,
                [status["entities"]["hashtags"] for status in self.twitter_data]
            ))
        ])

    # ============= Tweet Langs ============= #
    def _base_tweet_lang(self):
        self.tweet_lang_cnt = Counter([
            status["lang"]
            for status in self.twitter_data
        ])

    # ============= Relevant Words ============= #
    def _base_content_words(self):
        self.contents = [
            status["text"]
            for status in self.twitter_data
        ]

        if self.tokenizer_lang == "ko":
            contents_noun = list(reduce(
                lambda i, j: i + j,
                [
                    self.okt.nouns(self.okt.normalize(content))
                    for content in self.contents
                ]
            ))
        elif self.tokenizer_lang == "en":
            contents_toks = list(reduce(
                lambda i, j: i + j,
                [
                    nltk.word_tokenize(str(clean_content))
                    for clean_content in self.contents
                ]
            ))
            contents_noun = [word for (word, pos) in nltk.pos_tag(contents_toks) if pos[:2] == 'NN']
        else:
            raise ValueError("not supported language")

        self.contents_noun_cnt = Counter(contents_noun)

    def stat_content_words(self):
        top10_nouns = sorted(self.contents_noun_cnt.items(), key=lambda i: i[1], reverse=True)[:10]
        print("==================================")
        print("Most Frequent Nouns top-k (k=10)")
        pprint(top10_nouns)
        print("==================================")




if __name__ == '__main__':
    data_names = ["애플페이", "OpenAI"]

    for data_name in data_names:
        basic_stat = ContentStat(data_name=data_name)
        with open(f"{data_name}-stat.txt", 'w') as sys.stdout:
