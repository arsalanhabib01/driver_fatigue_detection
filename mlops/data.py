import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from sklearn.model_selection import GroupShuffleSplit
from utils import load_pickle
import os

dataset_loc = "DATAPATH"

def load_data(dataset_loc: str, subset: str = 'train', participant: int = 0, num_samples: int = None) -> Dataset:
    """Load data from source into a Ray Dataset.

    Args:
        dataset_loc (str): Location of the dataset.
        num_samples (int, optional): The number of samples to load. Defaults to None.

    Returns:
        Dataset: Our dataset represented by a Ray Dataset.
    """
    fpath = os.path.join(dataset_loc, subset+"_participant"+str(participant)+".pkl")
    data, targets, groups = load_pickle(fpath)
    ds = pd.DataFrame({'groups': groups, 'targets': targets})
    ds['id'] = ds.index.values.tolist()
    ds['features'] = ds['id'].apply(lambda r: data[r])
    ds = ray.data.from_pandas(ds)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds


def group_split(
        ds: Dataset,
        test_size: float,
        group_column: str = 'groups',
        seed: int = 1234,
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and test splits with equal
    amounts of data points from each class in the column we
    want to stratify on.

    Args:
        ds (Dataset): Input dataset to split.
        test_size (float): Proportion of dataset to split for test set.
        group_column (str, optional): Name of column to group and split on.
        seed (int, optional): seed for shuffling. Defaults to 1234.

    Returns:
        Tuple[Dataset, Dataset]: the stratified train and test datasets.
    """
    train_split =  1 - test_size
    gss = GroupShuffleSplit(n_splits=1, train_size=train_split, random_state=seed)
    df_pandas = ds.to_pandas()
    train_index, valid_index  = next(gss.split(np.vstack(df_pandas['features']), df_pandas['targets'], df_pandas[group_column]))
    # Shuffle each split (required)
    train_ds = df_pandas.iloc[train_index]
    test_ds = df_pandas.iloc[valid_index]
    return ray.data.from_pandas(train_ds), ray.data.from_pandas(test_ds)


def rearranged(batch: Dict) -> Dict:
    """Tokenize the text input in our batch using a tokenizer.

    Args:
        batch (Dict): batch of data with the text inputs to tokenize.

    Returns:
        Dict: batch of data with the results of tokenization (`input_ids` and `attention_mask`) on the text inputs.
    """

    xdnnt = np.reshape(np.vstack(batch["features"]), (-1, 4000, 4))
    xdnnt = xdnnt[:, ::5]
    xdnnt = xdnnt.transpose(0, 2, 1)
    return dict(features=xdnnt, groups=np.array(batch['groups']), targets=np.array(batch["targets"]))


def preprocess(df: pd.DataFrame, class_to_index: Dict) -> Dict:
    """Preprocess the data in our dataframe.

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.
        class_to_index (Dict): Mapping of class names to indices.

    Returns:
        Dict: preprocessed data (ids, masks, targets).
    """
    df = df.drop(columns=["id"], errors="ignore")  # clean dataframe
    df = df[["features", "targets", "groups"]]  # rearrange columns
    df["targets"] = df["targets"].map(class_to_index)  # label encoding
    outputs = rearranged(df)
    return outputs


class CustomPreprocessor(Preprocessor):
    """Custom preprocessor class."""
    def _fit(self, ds):
        targets = ds.unique(column="targets")
        self.class_to_index = {target: i for i, target in enumerate(targets)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
    def _transform_pandas(self, batch):  # could also do _transform_numpy
        return preprocess(batch, class_to_index=self.class_to_index)
