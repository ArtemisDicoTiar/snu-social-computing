import sys
from pprint import pprint
from typing import Tuple, Union, List

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from week11_supervised.dataloader import DataLoader


class SupervisedModel:
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
                 model: SupervisedModel = None,
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


if __name__ == '__main__':
    dataloader = DataLoader()

    model_params = {
        "gaussian": {},
        "logistic_regression": {},
        "support_vector": {
            "C": [0.1, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4],
            "kernel": ["linear", "rbf"],
            "gamma": [0.1, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
        },
        "decision_tree": {
            "max_depth": [3, None]
        },
        "random_forest": {
            "n_estimators": [100, 200]
        },
        "knn": {
            "n_neighbors": [1, 3, 5]
        },
        # "gradient_boost": {
        #     "learning_rate": [0.005, 0.008, 0.1, 0.15],
        #     "n_estimators": [80, 100, 150, 200],
        #     "max_depth": [2, 3, 4],
        #     "min_samples_split": [2, 3, 4]
        # }
    }
    y_preds = []
    with open(f"base-run-stat.txt", 'w') as sys.stdout:
        for model_name, params in model_params.items():
            pprint(f"{model_name=}")
            model = SupervisedModel(model_name)
            trainer = SupervisedTrainer(data=dataloader, model=model)
            trainer.train(params)
            trainer.test()
            trainer.metrics()
            y_preds.append(trainer.y_pred)
            print()

        for v in np.arange(0.1, 0.9, 0.1):
            print(f"ensemble: {v}")
            ensemble = (np.stack(y_preds).mean(axis=0) > v).astype(int)
            trainer = SupervisedTrainer(data=dataloader, model=SupervisedModel("gaussian"))
            trainer.metrics(ensemble, name=f"ensemble_{v}")
            print()
