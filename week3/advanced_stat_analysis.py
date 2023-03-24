import json
import sys
from collections import Counter
from datetime import datetime
from functools import reduce
from pprint import pprint
from typing import List, Optional, Any

import numpy as np
import statsmodels.api as sm

import pandas as pd
from konlpy.tag import Okt
from matplotlib import pyplot as plt
from numpy import log2

from pathlib import Path

# --- The directories --- #
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# For Apple Silicon
JVM_PATH = '/opt/homebrew/opt/openjdk@17/bin/java'


class AdvancedStat:
    def __init__(self, data_name: str, tokenizer_lang: str = "ko"):
        self.normalized_rt_rate = None
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
        self._base_retweets()
        self._base_analysis_data_df()

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
        self.is_hashtags = [
            1 if len(status["entities"]["hashtags"]) > 0 else 0
            for status in self.twitter_data
        ]
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

    # ============= RTd Tweets ============= #
    def _base_retweets(self):
        self.num_retweeted = [
            status["retweet_count"] if 'retweeted_status' in status else 0 for status in self.twitter_data
        ]

        # Want to be Retweeted? Large Scale Analytics on Factors Impacting Retweet in Twitter Network
        # Suh, 2010
        # similar to the paper
        # here, rate is calculated a bit different.
        self.normalized_rt_rate = [
            rt / t
            for rt, t in zip(self.num_retweeted, self.num_tweets_by_user)
        ]

    def plot_matrix_graph(self):
        df = pd.DataFrame(zip(
            # self.normalized_rt_rate,
            self.num_retweeted,
            self.num_tweets_by_user,
            self.num_followers,
            self.tweet_crated_as_unix_times,
            self.user_crated_as_unix_times
        ), columns=[
            # "norm_rt",
            "tweets", "twts_users", "followers", "twt_created", "usr_created"
        ]
        )
        pd.plotting.scatter_matrix(df, figsize=(12, 12), grid=True)
        plt.savefig(f"./images/{self.data_name}-matrix-graph.png")
        plt.close()

    def _base_analysis_data_df(self):
        dfX0 = pd.DataFrame(
            zip(
                [t - min(self.tweet_crated_as_unix_times) for t in self.tweet_crated_as_unix_times],
                [t - min(self.user_crated_as_unix_times) for t in self.user_crated_as_unix_times],
                self.is_hashtags,
                [
                    1 if len(status["entities"]["user_mentions"]) > 0 else 0
                    for status in self.twitter_data
                ],
                self.num_followers,
                self.num_tweets_by_user,
                [status["user"]["friends_count"] for status in self.twitter_data],
                [status["favorite_count"] for status in self.twitter_data],
            ),
            columns=[
                "twt_created",
                "user_created",
                "hashtag",
                "mention",
                "followers",
                "twts_user",
                "friends",
                "favorites"
            ],

        )
        self.dfX = sm.add_constant(dfX0)
        self.dfy = pd.DataFrame(self.num_retweeted, columns=["rt_cnt"])

    def stat_generalized_linear_model(self):
        # Want to be Retweeted? Large Scale Analytics on Factors Impacting Retweet in Twitter Network
        # Suh, 2010
        # section 6. -> predict the probability of retweet
        # similar but input params changed.

        model = sm.GLM(self.dfy, self.dfX)
        model_res = model.fit()
        print(model_res.summary())

    def stat_pca_plot_factor_map(self):
        pca = PCA(n_components=2)
        total_df = self.dfX
        total_df["rt_cnt"] = self.dfy["rt_cnt"]
        total_df = StandardScaler().fit_transform(total_df)

        pca.fit(total_df)

        # Get the PCA components (loadings)
        PCs = pca.components_
        fts = [
            "twt_created",
            "user_created",
            "hashtag",
            "mention",
            "followers",
            "twts_user",
            "friends",
            "favorites",
            "rt_cnt"
        ]
        pcs_df = pd.DataFrame(PCs[:, 1:], columns=fts).T
        print("================ PCA ================")
        print(pcs_df)

        # Use quiver to generate the basic plot
        fig = plt.figure(figsize=(5, 5))
        plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
                   PCs[0, :], PCs[1, :],
                   angles='xy', scale_units='xy', scale=1)

        # Add labels based on feature names (here just numbers)
        feature_names = fts
        for i, j, z in zip(PCs[1, :], PCs[0, :], feature_names):
            plt.text(j + 0.04 if j > 0 else j - 0.04, i + 0.04 if i > 0 else i - 0.04, z, ha='center', va='center')

        # Add unit circle
        circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
        plt.gca().add_artist(circle)

        # Ensure correct aspect ratio and axis limits
        plt.axis('equal')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])

        # Label axes
        plt.xlabel('PC 0')
        plt.ylabel('PC 1')

        # Done
        plt.savefig(f"./images/{self.data_name}-factormap.png")


if __name__ == '__main__':
    data_names = ["애플페이", "OpenAI"]

    for data_name in data_names:
        with open(f"{data_name}-adv-stat.txt", 'w') as sys.stdout:
            advanced_stat = AdvancedStat(data_name=data_name)
            advanced_stat.plot_matrix_graph()
            advanced_stat.stat_generalized_linear_model()
            advanced_stat.stat_pca_plot_factor_map()

