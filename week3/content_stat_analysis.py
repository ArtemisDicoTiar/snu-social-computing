import json
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from functools import reduce
from pprint import pprint
from string import punctuation
from typing import List, Union, Tuple, Any, Optional

import nltk
from konlpy.tag import Okt
from matplotlib import pyplot as plt
from nltk import PorterStemmer, sent_tokenize, word_tokenize

from pathlib import Path

# --- The directories --- #
from wordcloud import WordCloud

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# For Apple Silicon
JVM_PATH = '/opt/homebrew/opt/openjdk@17/bin/java'

# NLTK preparation
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download('universal_tagset')

from nltk.corpus import stopwords


class Tokenizer:
    twt_stop_words = ['.', ',', '--', '\'s', '?', '!', ')', '(', ':', '\'', '\'re', '"', '-', '}', '{', u'—', 'rt']

    def __init__(self, lang: str = "ko"):
        self.lang = lang

        if self.lang == "ko":
            self.ko_tokenizer = Okt(jvmpath=JVM_PATH)

            # https://www.ranks.nl/stopwords/korean
            with open("./korean_stopwords.txt", "r") as f:
                self.stopwords = [word.strip() for word in f.readlines()] + self.twt_stop_words

        if self.lang == "en":
            self.en_stemmers = PorterStemmer()
            self.stopwords = stopwords.words('english') + self.twt_stop_words

    # ============= Utils ============= #
    @staticmethod
    def _lower_strip(sentence: str) -> str:
        return sentence.lower().strip()

    def _remove_stopwords(self, terms: Union[str, List[str]]) -> Union[str, List[str]]:
        url_condition = lambda w: (w.startswith("t.co") or w.startswith("http") or w.startswith("https"))
        tweet_meta = lambda w: w.startswith("@") or w.startswith("#")
        if type(terms) == list:
            return [
                w
                for w in terms
                if w not in self.stopwords
                # removing urls
                if not url_condition(w)
                if not tweet_meta(w)
            ]

        if type(terms) == str:
            return terms \
                if (terms not in self.stopwords) and not url_condition(terms) and not tweet_meta(terms) \
                else None

    @staticmethod
    def _remove_punctuation(terms: Union[str, List[str]]) -> Union[str, List[str]]:
        if type(terms) == list:
            punc_removed = [word.strip(punctuation) for word in terms]
            return [word for word in punc_removed if word]

        if type(terms) == str:
            return terms if terms.strip(punctuation) else None

    def _regularize(self, terms: Union[str, List[str]]) -> Union[str, List[str]]:
        if self.lang == "en":
            return [
                t for t
                in self._remove_stopwords(
                    self._remove_punctuation([
                        self._lower_strip(t) for t in terms
                    ])
                )
                if t.strip()
            ]
        if self.lang == "ko":
            return [
                t for t
                in self._remove_stopwords(self._remove_punctuation(
                    terms
                ))
                if t.strip()
            ]

    def tokenize(self, sentence: str) -> List[str]:
        if self.lang == "en":
            return self._regularize(word_tokenize(sentence))

        if self.lang == "ko":
            return self._regularize(self.ko_tokenizer.morphs(sentence))

        raise ValueError("Wrong Language")

    def get_pos_tags(self, tokenized_terms: List[str]) -> List[Tuple[Union[str, List[str]], Any]]:
        if self.lang == "ko":
            return [self.ko_tokenizer.pos(t)[0] for t in tokenized_terms]

        if self.lang == "en":
            return nltk.pos_tag(tokenized_terms, tagset='universal')

    @staticmethod
    def _filter_pos_terms_by(pos_terms: List[Tuple[str, str]], term_type: str) -> List[str]:
        return [
            term[0]
            for term in pos_terms
            if term[1] == term_type
        ]

    def get_nouns(self, pos_terms: List[Tuple[str, str]]) -> List[str]:
        if self.lang == "ko":
            return self._filter_pos_terms_by(pos_terms, "Noun")
        if self.lang == "en":
            return self._filter_pos_terms_by(pos_terms, "NOUN")

    def get_verbs(self, pos_terms: List[Tuple[str, str]]) -> List[str]:
        if self.lang == "ko":
            return self._filter_pos_terms_by(pos_terms, "Verb")
        if self.lang == "en":
            return self._filter_pos_terms_by(pos_terms, "VERB")

    def get_adverbs(self, pos_terms: List[Tuple[str, str]]) -> List[str]:
        if self.lang == "ko":
            return self._filter_pos_terms_by(pos_terms, "Adverb")
        if self.lang == "en":
            return self._filter_pos_terms_by(pos_terms, "ADV")

    def get_adjectives(self, pos_terms: List[Tuple[str, str]]) -> List[str]:
        if self.lang == "ko":
            return self._filter_pos_terms_by(pos_terms, "Adjective")
        if self.lang == "en":
            return self._filter_pos_terms_by(pos_terms, "ADJ")


class ContentStat:
    def __init__(self, data_name: str, tokenizer_lang: str = "ko"):
        self.tokenized_terms = None
        self.pos_tags = None
        self.nouns = None
        self.verbs = None
        self.adverbs = None
        self.adjectives = None

        self.data_name = data_name
        self.file_name = f"{data_name}.json"

        self.twitter_data = self._load_twitter_json()

        self.tokenizer_lang = tokenizer_lang
        self.tokenizer = Tokenizer(self.tokenizer_lang)

        self._base_tweet_created()
        self._base_user_created()
        self._base_num_tweets_by_user()
        self._base_num_followers()
        self._base_hashtags()
        self._base_tweet_lang()
        self._base_tweet_content()

    # ============= Data Load ============= #
    def _load_twitter_json(self):
        with (DATA_DIR / self.file_name).open() as f:
            statuses = json.loads(f.read())
        return statuses

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

    # ============= Tweet Contents ============= #
    def _base_tweet_content(self):
        self.tweet_content = [
            status["text"]
            for status in self.twitter_data if status["lang"] == self.tokenizer_lang
        ]

    def stat_words(self):
        tokenized_terms = [self.tokenizer.tokenize(tweet) for tweet in self.tweet_content]
        pos_tags = [self.tokenizer.get_pos_tags(toks) for toks in tokenized_terms]
        nouns = [self.tokenizer.get_nouns(toks) for toks in pos_tags]
        verbs = [self.tokenizer.get_verbs(toks) for toks in pos_tags]
        adverbs = [self.tokenizer.get_adverbs(toks) for toks in pos_tags]
        adjectives = [self.tokenizer.get_adjectives(toks) for toks in pos_tags]

        self.tokenized_terms = tokenized_terms
        self.pos_tags = pos_tags
        self.nouns = nouns
        self.verbs = verbs
        self.adverbs = adverbs
        self.adjectives = adjectives

    def word_frequency(self, of: Optional[str] = ""):
        if of == "all":
            target = self.tokenized_terms
        elif of == "noun":
            target = self.nouns
        elif of == "verb":
            target = self.verbs
        elif of == "adverb":
            target = self.adverbs
        elif of == "adjective":
            target = self.adjectives
        else:
            raise ValueError("Wrong POS")

        target_words = list(reduce(lambda i, j: i + j, target))
        target_word_cnt = Counter(target_words)
        k = 10
        topk = sorted(target_word_cnt.items(), key=lambda i: i[1], reverse=True)[:k]
        print(f"========== TOP-{k} {of}s ==========")
        pprint(topk)

        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        wordcloud = WordCloud(font_path=font_path, max_font_size=40, max_words=1000)
        wordcloud.generate_from_frequencies(target_word_cnt)

        plt.figure(figsize=(10, 10), facecolor=None)
        plt.title(f"{self.data_name} {of}")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"./images/{self.data_name}-{of}.png")
        plt.close()


if __name__ == '__main__':
    data_names = ["애플페이", "OpenAI"]
    langs = ["ko", "en"]
    poss = ["all", "noun", "verb", "adverb", "adjective"]
    for data_name, lang in zip(data_names, langs):
        with open(f"{data_name}-word-stat.txt", 'w') as sys.stdout:
            basic_stat = ContentStat(data_name=data_name, tokenizer_lang=lang)
            basic_stat.stat_words()

            for pos in poss:
                basic_stat.word_frequency(pos)


