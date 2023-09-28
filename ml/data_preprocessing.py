import pandas as pd
import os
import glob
import random
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import pickle

TS_LENGTH = 4000

def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def split_sequence(sequence, n_steps, annotation, group):
    X, y, g = list(), list(), list()
    for i in range(0, sequence.shape[0], 100):
        end_ix = i + n_steps - 1
        if end_ix > sequence.shape[0]-1:
            break
        seq_x = sequence.loc[i:end_ix].to_numpy(copy=True, dtype='float', na_value=-1)
        X.append(seq_x)
        y.append(annotation)
        g.append(group)
    X.append(sequence.iloc[-n_steps:].to_numpy(copy=True, dtype='float', na_value=-1))
    y.append(annotation)
    g.append(group)
    return np.array(X), np.array(y), np.array(g)


def remove_cols(df):
    return df.drop(['Y', 'Participant', 'Frame'], axis=1)


def read_data(datapath):
    paths = glob.glob(os.path.join(datapath, "*.csv"))
    print(len(paths), 'files in total')
    dfs = [pd.read_csv(x) for x in paths]
    labels = [int(d.Y.unique()[0]) for d in dfs]
    groups = [int(d.Participant.unique()[0]) for d in dfs]
    # fill in the nan with -1 but ffill option can be used
    dfs = [remove_cols(d).fillna(-1) for d in dfs]
    return dfs, labels, groups


def create_dataset(dfs, labels, groups):
    Xs, ys, gs = list(), list(), list()
    for d,l,g in zip(dfs, labels, groups):
        if l == 5:
            continue
        X, y, groups = split_sequence(d, TS_LENGTH, l, g)
        Xs.append(X)
        ys.extend(y)
        gs.extend(groups)
    Xs = np.concatenate(Xs, axis=0)
    return Xs, np.array(ys), np.array(gs)



def run(raw_data_path: str, dest_path: str, participant_test: int):
    d, l, g = read_data(raw_data_path)
    x, y, gs = create_dataset(d, l, g)
    total = len(x)
    xtotal = np.reshape(x, (total,-1))
    indtest = np.where(gs == participant_test)
    le = LabelEncoder()
    ytotal = le.fit_transform(y)
    xtest = xtotal[indtest]
    ytest = ytotal[indtest]
    gtest = gs[indtest]
    mask = np.ones(xtotal.shape[0], dtype=bool)
    mask[indtest] = False
    xtrain = xtotal[mask]
    ytrain = ytotal[mask]
    gtrain = gs[mask]
    # create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)
    # save dictvectorizer and datasets
    dump_pickle(le, os.path.join(dest_path, "lab_encoder.pkl"))
    dump_pickle((xtrain, ytrain, gtrain), os.path.join(dest_path, "train_participant"+str(participant_test)+".pkl"))
    dump_pickle((xtest, ytest, gtest), os.path.join(dest_path, "test_participant"+str(participant_test)+".pkl"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        help="the location where the raw data was saved."
    )
    parser.add_argument(
        "--dest_path",
        help="the location where the resulting files will be saved."
    )
    parser.add_argument(
        "--participant_test", nargs='?', default=0, type=int,
        help="the participant id which should put in the test set."
    )
    args = parser.parse_args()

    run(args.raw_data_path, args.dest_path, args.participant_test)
