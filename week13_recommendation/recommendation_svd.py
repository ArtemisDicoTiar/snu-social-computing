import sys

import pandas as pd
import numpy as np

from week2_tweet_stats.basic_stat_analysis import DATA_DIR

import numpy as np
from sklearn.metrics import mean_squared_error

# https://big-dream-world.tistory.com/69
def get_rmse(R, P, Q, non_zeros):
    error = 0
    # 두개의 분해된 행렬 P와 Q.T의 내적 곱으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)

    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]

    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse


def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape
    # P와 Q 매트릭스의 크기를 지정하고 정규분포를 가진 랜덤한 값으로 입력합니다.
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    break_count = 0

    # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트 객체에 저장.
    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    # SGD기법으로 P와 Q 매트릭스를 계속 업데이트.
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구함
            eij = r - np.dot(P[i, :], Q[j, :].T)
            # Regularization을 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)
        if (step % 10) == 0:
            print("### iteration step : ", step, " rmse : ", rmse)

    return P, Q
if __name__ == '__main__':
    with open(f"output.txt", 'w') as sys.stdout:
        movies = pd.read_csv(DATA_DIR / "ml-latest-small" / "movies.csv")
        ratings = pd.read_csv(DATA_DIR / "ml-latest-small" / "ratings.csv")

        ratings = ratings[["userId", "movieId", "rating"]]
        ratings_matrix = ratings.pivot_table("rating", index="userId", columns="movieId")

        rating_movies = pd.merge(ratings, movies, on="movieId")
        ratings_movies_matrix = rating_movies.pivot_table("rating", index="userId", columns="title")

        Ks = [25, 50, 75, 100, 125, 150]
        for K in Ks:
            P, Q = matrix_factorization(ratings_movies_matrix.values, K=K, steps=100, learning_rate=0.01, r_lambda=0.01)
            pred_matrix = np.dot(P, Q.T)

            ratings_pred_matrix = pd.DataFrame(data=pred_matrix,
                                               index=ratings_movies_matrix.index,
                                               columns=ratings_movies_matrix.columns)
            print(ratings_pred_matrix.head(3))

            userId = 10
            user_rating = ratings_movies_matrix.loc[userId, :]
            already_seen = user_rating[user_rating > 0].index.tolist()
            movies_list = ratings_movies_matrix.columns.tolist()
            unseen_list = list(filter(lambda movie: movie not in already_seen, movies_list))
            print(unseen_list[:3])

            print(ratings_pred_matrix.loc[userId, unseen_list])

            print(ratings_pred_matrix.loc[userId, unseen_list].sort_values(ascending=False)[:10])

