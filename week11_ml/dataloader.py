from pathlib import Path
from typing import Union

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from week2_tweet_stats.basic_stat_analysis import DATA_DIR


class DataLoader:
    def __init__(self,
                 file_path: Union[str, Path] = DATA_DIR / "winequality-red.csv",
                 cross_validate: bool = True,
                 cv_k: int = 5,
                 cut: bool = True):
        df = pd.read_csv(file_path, sep=";")
        self.df = df

        # Dividing wine as good and bad by giving the limit for the quality
        bins = (2, 6.5, 8)
        if cut:
            group_names = ['bad', 'good']
            df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)

            # Now lets assign a labels to our quality variable
            label_quality = LabelEncoder()
            # Bad becomes 0 and good becomes 1
            df['quality'] = label_quality.fit_transform(df['quality'])

        y = df["quality"].values
        self.y = y
        X = df[filter(lambda col: col != "quality", df.columns.tolist())].values
        self.X = X

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Applying Standard scaling to get optimized result
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        self._train = (X_train, y_train)
        self._test = (X_test, y_test)

        self.cv_k_fold = KFold(n_splits=cv_k) if cross_validate else None

    def get_train_data(self):
        return self._train

    def get_test_data(self):
        return self._test

if __name__ == '__main__':
    dataloader = DataLoader()

    from pandas.plotting import scatter_matrix

    scatter_matrix(dataloader.df, alpha=0.2, figsize=(18, 18), diagonal='kde')
    plt.savefig("./images/dataset_desc.png")
    # plt.show()
    print()
