import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

import sys
from pprint import pprint
from typing import Tuple, Union, List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from scipy.cluster.hierarchy import dendrogram, linkage

from week11_ml.dataloader import DataLoader
from week2_tweet_stats.basic_stat_analysis import DATA_DIR


class UnsupervisedModel:
    def __init__(self, which: str = None, params=None):
        if params is None:
            params = {}

        assert which is not None

        self.model_list = {
            "gaussian": GaussianNB(**params),
            "logistic_regression": LogisticRegression(**params),
            "support_vector": SVC(**params, probability=True),
            "decision_tree": DecisionTreeClassifier(**params),
            "random_forest": RandomForestClassifier(**params),
            "knn": KNeighborsClassifier(**params),
            #
            "gradient_boost": GradientBoostingClassifier(**params)
        }
        self.model = self.model_list[which]

    def fit(self, data: Tuple[ndarray, ndarray]) -> None:
        self.model.fit(*data)

    def predict(self, feature: Union[ndarray, List]):
        return self.model.predict(feature)

    def score(self, data: Tuple[ndarray, ndarray]):
        return self.model.score(*data)



class SupervisedTrainer:
    def __init__(self,
                 model: UnsupervisedModel = None,
                 data: DataLoader = None):
        assert model is not None
        assert data is not None

        self.model_wrapper = model
        self.data = data

        self.cv = True if data.cv_k_fold is not None else False

        self.y_pred = None
        self.y_pred_proba = None

    def train(self, params):
        search = GridSearchCV(self.model_wrapper.model, params, cv=5, verbose=1)
        search.fit(*self.data.get_train_data())
        self.model_wrapper.model = search.best_estimator_
        print(f"{search.best_params_=}")

    def metrics(self, y_pred = None, name: str = None):
        y_real = self.data.get_test_data()[-1]
        if y_pred is None:
            fpr, tpr, _ = roc_curve(y_real, self.y_pred_proba)
            auc = roc_auc_score(y_real, self.y_pred_proba)
            plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
            plt.legend(loc=4)

            if name is None:
                plt.title(self.model_wrapper.model.__class__.__name__)
                plt.savefig(f"./images/{self.model_wrapper.model.__class__.__name__}.png")

            else:
                plt.title(name)
                plt.savefig(f"./images/{name}.png")

            plt.close()
        # plt.show()
        if y_pred is None:
            y_pred = self.y_pred

        conf_mat = confusion_matrix(y_real, y_pred)
        print(f"conf_mat=\n{conf_mat}")

        acc = accuracy_score(y_real, y_pred)
        prec = precision_score(y_real, y_pred)
        recall = recall_score(y_real, y_pred)
        print(f"{acc=}")
        print(f"{prec=}")
        print(f"{recall=}")

    def test(self):
        self.model_wrapper.score(self.data.get_test_data())
        self.y_pred = self.model_wrapper.predict(self.data.get_test_data()[0])
        self.y_pred_proba = self.model_wrapper.model.predict_proba(self.data.get_test_data()[0])[::, 1]

def pca(x, y, save_fname: str = None):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(y, columns=["quality"])], axis=1)

    # print(f"pca result=\n{finalDf}")
    print(f"{pca.explained_variance_ratio_=}")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Wine PCA', fontsize=20)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ss = ax.scatter(finalDf['principal component 1'], finalDf['principal component 2'],
                    alpha = 0.7, c = finalDf['quality'], cmap = 'jet')
    plt.colorbar(ss)
    ax.grid()
    if save_fname is not None:
        plt.savefig(save_fname)
        plt.close()
    else:
        plt.show()



def tsne(x, y, save_fname: str = None):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
    low_dim_embs = tsne.fit_transform(x)
    labels = y
    fig = plt.figure(figsize=(16, 16))  # in inches
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Wine TSNE', fontsize=20)
    ss = ax.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1],
                    alpha=0.7, c=labels, cmap='jet')
    plt.colorbar(ss)
    ax.grid()

    if save_fname is not None:
        plt.savefig(save_fname)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    for color in ['red', 'white']:
        dataloader = DataLoader(file_path=DATA_DIR / f"winequality-{color}.csv",cut=False)

        Xs = StandardScaler().fit_transform(dataloader.X)

        ys = dataloader.y
        sum_of_squared_distances = []
        ks = range(0, 30)
        for k in ks:
            labels = ys
            if k != 0:
                kmeans = KMeans(n_clusters=k, n_init='auto')
                kmeans.fit(Xs)

                sum_of_squared_distances.append(kmeans.inertia_)
                labels = kmeans.labels_

            pca(Xs, labels, save_fname=f"./images/unsupervised/{color}/pca-kmeans-k={k}.png")
            tsne(Xs, labels, save_fname=f"./images/unsupervised/{color}/tsne-kmeans-k={k}.png")
            print(f"{k} Done...")

        plt.plot(ks[1:], sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.savefig(f"./images/unsupervised/{color}/optimal_k.png")
        plt.close()
        plt.show()

    exit()
    red_dataloader = DataLoader(file_path=DATA_DIR / f"winequality-red.csv", cut=False)
    white_dataloader = DataLoader(file_path=DATA_DIR / f"winequality-white.csv", cut=False)

    Xs = StandardScaler().fit_transform(np.concatenate([red_dataloader.X, white_dataloader.X], axis=0))

    # ys = np.concatenate([red_dataloader.y, white_dataloader.y], axis=0)
    ys = np.array([0 for _ in range(len(red_dataloader.y))] + [1 for _  in range(len(white_dataloader.y))])
    sum_of_squared_distances = []
    ks = range(0, 1)
    for k in ks:
        labels = ys
        if k != 0:
            kmeans = KMeans(n_clusters=k, n_init='auto')
            kmeans.fit(Xs)

            sum_of_squared_distances.append(kmeans.inertia_)
            labels = kmeans.labels_

        pca(Xs, labels, save_fname=f"./images/unsupervised/total/color-pca-kmeans-k={k}.png")
        tsne(Xs, labels, save_fname=f"./images/unsupervised/total/color-tsne-kmeans-k={k}.png")
        print(f"{k} Done...")

    plt.plot(ks[1:], sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(f"./images/unsupervised/total/color-optimal_k.png")
    plt.close()