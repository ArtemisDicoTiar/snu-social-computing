import json
import sys
from collections import Counter
from datetime import datetime
from functools import reduce
from pprint import pprint
from typing import List, Optional, Any

import matplotlib
import nltk
from konlpy.tag import Okt
from matplotlib import pyplot as plt
from numpy import log2

from pathlib import Path

# --- The directories --- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


# For Apple Silicon
JVM_PATH = '/opt/homebrew/opt/openjdk@17/bin/java'


class BasicStat:
    def __init__(self, data_name: str, tokenizer_lang: str = "ko"):
        self.data_name = data_name
        self.file_name = f"{data_name}.json"

        self.twitter_data = self._load_twitter_json()

        self.tokenizer_lang = tokenizer_lang
        self.okt = Okt(jvmpath=JVM_PATH)

        self._base_tweet_created()
        self._base_user_created()
        self._base_num_tweets_by_user()
        self._base_num_followers()
        self._base_hashtags()
        self._base_tweet_lang()
        self._base_user_lang()
        self._base_content_words()

    # ============= Data Load ============= #
    def _load_twitter_json(self):
        with (DATA_DIR / self.file_name).open() as f:
            statuses = json.loads(f.read())
        return statuses

    # ============= Utils ============= #
    @staticmethod
    def _print_stat(
            which: str = "something",
            items: Optional[list] = None,
            values: Optional[List[Any]] = None
    ):
        if items is None and values is None:
            raise ValueError("Either of two input must be given.")

        if values:
            min_val, max_val, mean_val, med_val = values

        if items:
            min_val = min(items)
            max_val = max(items)
            mean_val = sum(items) / len(items)

            sorted_items = sorted(items)
            if len(items) % 2 == 0:
                med_val = (sorted_items[len(items) // 2 - 1] + sorted_items[len(items) // 2]) / 2
            else:
                med_val = sorted_items[len(items) // 2]

        print("==================================")
        print(f"STATISTICS of {which}")
        print(f"min: \t{min_val}")
        print(f"max: \t{max_val}")
        print(f"mean: \t{mean_val}")
        print(f"median: {med_val}")
        print("==================================")

    def _stat_time(self, which: str, unix_time_data: List[float]):
        min_unix = min(unix_time_data)
        max_unix = max(unix_time_data)
        mean_unix = sum(unix_time_data) / len(unix_time_data)
        sorted_unix_time = sorted(unix_time_data)
        if len(unix_time_data) % 2 == 0:
            med_unix = (sorted_unix_time[len(unix_time_data) // 2 - 1] + sorted_unix_time[len(unix_time_data) // 2]) / 2
        else:
            med_unix = sorted_unix_time[len(unix_time_data) // 2]

        min_time = datetime.utcfromtimestamp(min_unix).strftime("%a %b %d %H:%M:%S %z %Y")
        max_time = datetime.utcfromtimestamp(max_unix).strftime("%a %b %d %H:%M:%S %z %Y")
        mean_time = datetime.utcfromtimestamp(mean_unix).strftime("%a %b %d %H:%M:%S %z %Y")
        med_time = datetime.utcfromtimestamp(med_unix).strftime("%a %b %d %H:%M:%S %z %Y")

        self._print_stat(
            which=which,
            values=[min_time, max_time, mean_time, med_time]
        )

    @staticmethod
    def _plot_hist(savename: str,
                   data: List[Any],
                   title: str = "",
                   x_axis: str = "",
                   y_axis: str = "",
                   loglog: bool = False
                   ):
        plt.figure(figsize=(12, 6))
        plt.hist(data, bins=200, rwidth=0.8)
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.grid()
        plt.savefig(f"./images/{savename}")
        plt.close()

        if loglog:
            plt.figure(figsize=(8, 6))
            plt.hist(log2(list(filter(lambda i: i > 0, data))), log=True, bins=200, rwidth=0.5)
            plt.title(title)
            plt.xlabel(f"{x_axis} (log)")
            plt.ylabel(f"{y_axis} (log)")
            plt.grid()
            plt.savefig(f"./images/loglog-{savename}")
            plt.close()

    def _plot_time_hist(self,
                        savename: str,
                        dates: List[datetime],
                        title: str = "",
                        x_axis: str = "Date/Time",
                        y_axis: str = ""
                        ):
        self._plot_hist(savename=savename, data=dates, title=title, x_axis=x_axis, y_axis=y_axis)

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

    def stat_tweet_created(self) -> None:
        self._stat_time(which="Tweet Created Datetime (UTC)", unix_time_data=self.tweet_crated_as_unix_times)

    def plot_tweet_created(self):
        self._plot_time_hist(
            savename=f"{self.data_name}-tweet-created.jpg",
            dates=self.tweet_created,
            title="When did the tweets post (for the given tweets)?",
            y_axis="Number of Tweets"
        )

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

    def stat_user_created(self):
        self._stat_time(
            which="User Created Datetime (UTC)",
            unix_time_data=self.user_crated_as_unix_times
        )

    def plot_user_created(self):
        self._plot_time_hist(
            savename=f"{self.data_name}-user-created.jpg",
            dates=self.user_created,
            title="When did the accounts created (for the given tweets)?",
            y_axis="Number of Users"
        )

    # ============= Tweets <- User ============= #
    def _base_num_tweets_by_user(self):
        self.num_tweets_by_user = [
            status["user"]["statuses_count"]
            for status in self.twitter_data
        ]

    def stat_num_tweets_by_user(self):
        self._print_stat(which="Number of Tweets By each user", items=self.num_tweets_by_user)

    def plot_num_tweets_by_user(self):
        self._plot_hist(savename=f"{self.data_name}-num-tweets-by-user.jpg", data=self.num_tweets_by_user,
                        title="How many Tweets posted by each user (from the given tweets)?",
                        x_axis="Number of Tweets",
                        y_axis="Counts",
                        loglog=True)

    # ============= Followers ============= #
    def _base_num_followers(self):
        self.num_followers = [
            status["user"]["followers_count"]
            for status in self.twitter_data
        ]

    def stat_num_followers(self):
        self._print_stat(which="Number of Followers", items=self.num_followers)

    def plot_num_followers(self):
        self._plot_hist(savename=f"{self.data_name}-num-followers.jpg", data=self.num_followers,
                        title="How many Followers (users from the given tweets)?",
                        x_axis="Number of Followers",
                        y_axis="Counts",
                        loglog=True)

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

    def stat_hashtags(self):
        top10_hashtags = sorted(self.hashtags_cnt.items(), key=lambda i: i[1], reverse=True)[:10]
        print("==================================")
        print("Hashtags top-k (k=10)")
        pprint(top10_hashtags)
        print("==================================")

    # ============= Tweet Langs ============= #
    def _base_tweet_lang(self):
        self.tweet_lang_cnt = Counter([
            status["lang"]
            for status in self.twitter_data
        ])

    def stat_tweet_lang(self):
        top10_tweet_langs = sorted(self.tweet_lang_cnt.items(), key=lambda i: i[1], reverse=True)[:10]
        print("==================================")
        print("Tweet Language top-k (k=10)")
        pprint(top10_tweet_langs)
        print("==================================")

    # ============= User Langs ============= #
    def _base_user_lang(self):
        self.user_lang_cnt = Counter([
            status["user"]["lang"]
            for status in self.twitter_data
        ])

    def stat_user_lang(self):
        top10_user_langs = sorted(self.user_lang_cnt.items(), key=lambda i: i[1], reverse=True)[:10]
        print("==================================")
        print("User Language top-k (k=10)")
        pprint(top10_user_langs)
        print("==================================")

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

    # ============= RTd Tweets ============= #
    def stat_retweets(self, ):
        retweeted_tweets = [status for status in self.twitter_data if 'retweeted_status' in status]
        print(f"중복 포함한 전체 RT 개수: {len(retweeted_tweets)}")

        # remove duplicates with dictionary
        retweets_id = {status["retweeted_status"]["id"] for status in retweeted_tweets}
        retweets = {
            status["retweeted_status"]["id"]: (
                status['retweet_count'], status['retweeted_status']['user']['screen_name'], status['text'])
            for status in self.twitter_data if 'retweeted_status' in status
        }
        print(f"Tweet ID로 중복 제거한 RT 개수: {len(retweets)}")

        duplicate_content_removed_retweets = {
            retweet[1][2]: (retweet[1][0], retweet[1][1])
            for retweet in retweets.items()
        }
        print(f"내용 중복 제거한 RT 개수: {len(duplicate_content_removed_retweets)}")

        top10_rts = list(map(
            lambda i: {
                "User": i[1][1],
                "Tweet": i[0],
                "RT count": i[1][0]
            },
            sorted(
                duplicate_content_removed_retweets.items(),
                key=lambda rt: rt[1][0],
                reverse=True
            )[:10]
        ))
        print("==================================")
        print("top-k retweeted Tweets (k=10)")
        pprint(top10_rts)
        print("==================================")


if __name__ == '__main__':
    data_names = ["애플페이", "OpenAI"]

    for data_name in data_names:
        basic_stat = BasicStat(data_name=data_name)
        with open(f"{data_name}-stat.txt", 'w') as sys.stdout:
            basic_stat.stat_tweet_created()
            basic_stat.plot_tweet_created()

            basic_stat.stat_user_created()
            basic_stat.plot_user_created()

            basic_stat.stat_num_tweets_by_user()
            basic_stat.plot_num_tweets_by_user()

            basic_stat.stat_num_followers()
            basic_stat.plot_num_followers()

            basic_stat.stat_hashtags()

            basic_stat.stat_tweet_lang()
            basic_stat.stat_user_lang()

            basic_stat.stat_content_words()

            basic_stat.stat_retweets()
