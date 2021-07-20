import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import MovieLensDataset as mld


def read_data_ml100k():
    data_path = "MovieLens_Data\\ml-100k\\u.data"
    labels = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(data_path, '\t', names=labels, engine='python')
    return data


def train_test_split(data, split_mode='random', test_ratio=0.1):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    if split_mode == 'seq-aware':

        for (user_id, records) in list(data.groupby(["user_id"])):
            test_index_del = records.index[records["timestamp"].argmax()]
            test_df = test_df.append(data.iloc[test_index_del])
            train_df = train_df.append(records.drop(test_index_del).sort_values("timestamp"))

        test_df = pd.DataFrame(test_df).reset_index(drop=True)
        train_df = pd.DataFrame(train_df).reset_index(drop=True)

    else:
        mask = [True if x == 1 else False for x in np.random.uniform(0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_df, test_df = data[mask], data[neg_mask]

    return train_df, test_df


def load_data(data, movie_num, feed_back="explicit"):
    users = [int(x - 1) for x in data["user_id"].tolist()]
    items = [int(y - 1) for y in data["item_id"].tolist()]
    inter = torch.zeros(movie_num, data["user_id"].nunique())
    scores = data["rating"].tolist()

    if feed_back == "explicit":
        for uid, iid, rate in zip(users, items, scores):
            inter[iid, uid] = rate
    else:
        pass

    return users, items, scores, inter


def split_and_load_data(device, split_mode="seq-aware", feed_back="explicit", test_ratio=0.1, batch_size=256,
                        encoded=False):
    data = read_data_ml100k()
    train_d, test_d = train_test_split(data, split_mode=split_mode, test_ratio=test_ratio)

    train_user, train_movies, train_scores, _ = load_data(train_d, 1682, feed_back)
    test_user, test_movies, test_scores, _ = load_data(test_d, 1682, feed_back)

    train_set = mld.MovieLensDataset(train_user, train_movies, train_scores, ohencoded=encoded, device=device)
    test_set = mld.MovieLensDataset(test_user, test_movies, test_scores, device=device, ohencoded=encoded)

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=False)

    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    return 943, 1682, train_dl, test_dl
