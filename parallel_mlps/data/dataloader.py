import logging
import pickle
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.datasets
import torch
import torch.utils.data
from scipy import stats
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils import shuffle

# from data import datautils


class Dataloader:
    def __init__(
        self,
        dataset_name,
        n_splits,
        feature_range=(0, 1),
        random_state=None,
        verbose=False,
        normalize=True,
        standardize=True,
        data_home=None,
        log=None,
        one_hot_encode_output=True,
    ):
        """Initialize the Dataloader object.

        Args:
            dataset_name (string): datasetname(version). ex: iris(2) will be iris dataset and version 2.
            n_splits ([type]): [description]
            feature_range (tuple, optional): [description]. Defaults to (0, 1).
            random_state ([type], optional): [description]. Defaults to None.
            verbose (bool, optional): [description]. Defaults to True.
            normalize (bool, optional): [description]. Defaults to True.
            standardize (bool, optional): [description]. Defaults to True.
            data_home ([type], optional): [description]. Defaults to None.
        """
        self.dataset_name = dataset_name
        matches = re.search(r"(.+)\((.+?)\)", dataset_name)
        if matches:
            self.dataset_version = int(matches.group(2))
            self.dataset_name = matches.group(1)
        else:
            self.dataset_version = 1
        self.data_home = data_home
        self.n_splits = n_splits
        self.feature_range = feature_range
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.one_hot_encoder = OneHotEncoder(categories="auto", sparse=False)
        self.label_encoder = LabelEncoder()
        self.verbose = verbose
        self.normalize = normalize
        self.standardize = standardize
        self.x = None
        self.y = None
        self.in_features = None
        self.out_features = None
        self.n_samples = None
        self.log = log or logging.getLogger(__name__)
        self.one_hot_encode_output = one_hot_encode_output

        self.__load_data()
        if verbose:
            self.__describe_dataset()
        self.__initialize_kfold_splits()

    def get_imbalance_score(self):
        """
        Imbalance score_strategy meaning 1 = imbalance, 0 = balance
        Returns: Imbalance score_strategy between 0-1
        """
        _, counts = self.get_classes_counts()
        s = 1 - stats.entropy(counts) / np.log(len(counts))

        return s

    def get_classes_counts(self):
        classes, counts = np.unique(self.y, return_counts=True)
        return classes, counts

    def __initialize_kfold_splits(self):
        self.skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

    def __load_data(self):
        self.log.info(f"Loading {self.dataset_name}")
        cache_dir = Path(self.data_home) / "cached" / self.dataset_name
        cache_file = cache_dir / f"{self.dataset_name}_v{self.dataset_version}.dumped"
        self.log.info(f"cache_dir: {cache_dir.absolute()}")
        self.log.info(f"cache_file: {cache_file.absolute()}")

        Path.mkdir(cache_dir, parents=True, exist_ok=True)

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                self.openml_data = pickle.load(f)
        else:
            self.openml_data = sklearn.datasets.fetch_openml(
                name=self.dataset_name,
                version=self.dataset_version,
                data_home=self.data_home,
            )
            with open(cache_file, "wb") as f:
                pickle.dump(self.openml_data, f)

        if isinstance(self.openml_data.data, scipy.sparse.csr.csr_matrix):
            self.openml_data.data = np.asarray(self.openml_data.data.todense())

        # Removing rows with nans
        nan_rows = np.isnan(self.openml_data.data).any(axis=1)
        self.openml_data.data = self.openml_data.data[~nan_rows]
        self.openml_data.target = self.openml_data.target[~nan_rows]

        x_values = self.openml_data.data
        if not isinstance(x_values, np.ndarray):
            cat_columns = x_values.select_dtypes(["category"]).columns
            if len(cat_columns) > 0:
                x_values[cat_columns] = x_values[cat_columns].apply(
                    lambda x: x.cat.codes
                )
            x_values = x_values.values
        try:
            self.x = x_values.astype(np.float32)
        except ValueError as e:
            if type(x_values[0][0]) == str:
                ord_encoder = OrdinalEncoder()
                ord_encoder.fit(x_values)
                self.x = ord_encoder.transform(x_values)
            else:
                raise e

        self.y = self.label_encoder.fit_transform(self.openml_data.target)
        self.feature_names = self.openml_data.feature_names
        self.target_names = self.openml_data.target_names

        if self.dataset_name == "mnist_784":
            self.fixed_x_test = self.x[60000:]
            self.fixed_y_test = self.y[60000:][:, None]
            self.x = self.x[:60000]
            self.y = self.y[:60000]

        if type(self.x) == scipy.sparse.csr.csr_matrix:
            self.x = self.x.toarray()

        if self.y.ndim == 1:
            self.y = self.y[:, None]

        self.in_features = self.x.shape[-1]
        self.out_features = len(np.unique(self.y))
        self.n_samples = len(self.y)

        self.one_hot_encoder.fit(self.y)

    def __describe_dataset(self):
        self.log.info(self.openml_data.DESCR)
        self.log.info(self.openml_data.details)
        self.log.info(
            f"number of samples per label:\n{np.array(np.unique(self.y, return_counts=True))}."
        )
        self.log.info(scipy.stats.describe(self.x))
        self.log.info(scipy.stats.describe(self.y))

    def normalize_data(self, data):
        new_data = deepcopy(data)
        if isinstance(self.feature_range, tuple):
            self.min_max_scaler = MinMaxScaler(self.feature_range)
        else:
            self.min_max_scaler = StandardScaler()

        self.min_max_scaler.fit(data["train"]["data"])
        new_data["train"]["data"] = self.min_max_scaler.transform(data["train"]["data"])
        new_data["test"]["data"] = self.min_max_scaler.transform(data["test"]["data"])

        if "val" in data:
            new_data["val"]["data"] = self.min_max_scaler.transform(data["val"]["data"])

        return new_data

    def standardize_data(self, data):
        new_data = deepcopy(data)

        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(data["train"]["data"])
        new_data["train"]["data"] = self.standard_scaler.transform(
            data["train"]["data"]
        )
        new_data["test"]["data"] = self.standard_scaler.transform(data["test"]["data"])

        if "val" in data:
            new_data["val"]["data"] = self.standard_scaler.transform(
                data["val"]["data"]
            )

        return new_data

    def get_smallest_n_indices(self, a, N, maintain_order=True):
        idx = np.argpartition(a.ravel(), N)[:N]
        if maintain_order:
            idx = idx[a.ravel()[idx].argsort()]
        return np.stack(np.unravel_index(idx, a.shape)).T

    def get_smallest_sum(self, distances, considered_idxs, class_idxs):
        # Pega os n_splits exemplos de menor distancia entre si para fazer parte dos splits
        sum_distances = distances[:, considered_idxs].sum(1)
        sum_distances[~considered_idxs] = np.inf

        # Usando (i) apenas os indices nao utilizados anteriormente e (ii) apenas da classe k.
        # Colocando np.inf para forcar a ida dos indices que nao fazem sentido ficar na ultima posicao do sort.

        # # indices ja utilizados
        # sum_distances[used_idxs] = np.inf
        # # indices das outras classes
        # sum_distances[~idx_k] = np.inf

        # Pega o sample pivo, que tem a menor distancia para todos os outros samples da mesma classe
        pivot_idx = np.argpartition(sum_distances, 0)[0]
        return pivot_idx

    def get_distances(self, distance_name):
        x = self.x
        if distance_name is None:
            distance_name = "euclidean"

        cache_dir = Path(self.data_home) / "cached" / self.dataset_name
        cache_file = (
            cache_dir
            / f"{self.dataset_name}_v{self.dataset_version}_distances_{distance_name}.npy"
        )

        Path.mkdir(cache_dir, parents=True, exist_ok=True)

        if Path(cache_file).exists():
            distances = np.load(cache_file)
        else:
            if self.standardize:
                standard_scaler = StandardScaler()
                x = standard_scaler.fit_transform(self.x)
            if self.normalize:
                if self.feature_range:
                    min_max_scaler = MinMaxScaler(self.feature_range)
                else:
                    min_max_scaler = MinMaxScaler()
                x = min_max_scaler.fit_transform(self.x)
            # if distance_name == 'correlation' and torch.cuda.is_available():
            #     # distances = cuda_distances.gpu_dist_matrix(x, 'correlation')
            #     distances = datautils.torch_pairwise_correlation(x)
            # else:
            distances = distance.squareform(
                distance.pdist(x, metric=distance_name)
            ).astype(np.float32)
            np.save(cache_file, distances)

        return distances

    def sbss(self, distance_name, n_classes):
        distances = self.get_distances(distance_name)

        used_indexes = np.zeros(len(self.y)).astype(np.bool)
        # each append will have size = n_split
        folds_list = [
            [] for _ in range(self.n_splits)
        ]  # array with k rows and n_samples cols. Each column belongs to the same class
        splits = np.arange(self.n_splits)

        fold_col_lb = []

        for k in range(n_classes):
            # lb_samples[k] = []
            idx_k = self.y.squeeze() == k

            to_split_idxs_lb = idx_k.sum()

            while to_split_idxs_lb >= self.n_splits:
                ## Pegando o elemnto pivor, aquele que possui a menor distancia dele para os indicies ainda nao utilizados que fazem parte da classe k
                # Usando (i) apenas os indices nao utilizados anteriormente e (ii) apenas da classe k.
                considered_idxs = (~used_indexes) & idx_k
                # Pega os n_splits exemplos de menor distancia entre si para fazer parte dos splits
                sum_distances = np.nansum(distances[:, considered_idxs], axis=1)
                # Colocando np.inf para forcar a ida dos indices que nao fazem sentido ficar na ultima posicao do sort.
                sum_distances[~considered_idxs] = np.inf
                # Pega o sample pivo, que tem a menor distancia para todos os outros samples da mesma classe
                pivot_idx = np.argpartition(sum_distances, 0)[0]

                used_indexes[pivot_idx] = True
                # lb_samples[k].append(pivot_idx)

                nearby_samples = [pivot_idx]

                for split_idx in splits[1:]:
                    # Distance between all the elements and
                    sum_distances = np.nansum(distances[:, nearby_samples], axis=1)
                    # Ignorando indices que ja foram usados ou que nao sao da classe k
                    sum_distances[~considered_idxs] = np.inf
                    # Ignorando o proprio pivo, ja que nao faz sentido compara a distancia dele com ele mesmo
                    sum_distances[pivot_idx] = np.inf

                    # Get the smallest value index
                    closest_sample_idx = np.argpartition(sum_distances, 0)[0]
                    # lb_samples[k].append(closest_sample_idx)
                    nearby_samples.append(closest_sample_idx)

                    used_indexes[closest_sample_idx] = True

                    # Disconsidering this idxs for the next iterations
                    considered_idxs[closest_sample_idx] = False

                fold_col_lb.append(k)

                np.random.shuffle(nearby_samples)

                for split_idx in splits:
                    folds_list[split_idx].append(nearby_samples[split_idx])

                to_split_idxs_lb = to_split_idxs_lb - self.n_splits
        # with open('/tmp/lb_samples.pk', 'wb') as f:
        #     pickle.dump(lb_samples, f)

        folds = np.array(folds_list)
        fold_col_lb = np.array(fold_col_lb)

        return folds, fold_col_lb

    def get_splits_iter_regions(
        self, validation_rate_from_train=None, distance_name=None
    ):
        np.random.seed(self.random_state)
        # Default checking from scikitlearn kfold
        _, y_idx, y_inv = np.unique(self.y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            self.log.warning(
                (
                    "The least populated class in y has only %d"
                    " members, which is less than n_splits=%d."
                    % (min_groups, self.n_splits)
                ),
                UserWarning,
            )
        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        # test_folds = np.empty(len(self.y), dtype='i')

        # My code starts here...
        data = {"train": {}, "test": {}}

        cache_dir = Path(self.data_home) / "cached" / self.dataset_name
        cache_file = (
            cache_dir
            / f"{self.dataset_name}_v{self.dataset_version}_sbss_{distance_name}.npz"
        )

        Path.mkdir(cache_dir, parents=True, exist_ok=True)

        if Path(cache_file).exists():
            npzfiles = np.load(cache_file)
            folds = npzfiles["folds"]
            fold_col_lb = npzfiles["fold_col_lb"]
        else:
            # if distance_name == "correlation" and self.dataset_name == "mnist_784":
            #     folds, fold_col_lb = datautils.gpu_sbss(
            #         self.x, self.y, self.n_splits, similarity_name=distance_name
            #     )
            # else:
            #     folds, fold_col_lb = self.sbss(distance_name, n_classes)
            folds, fold_col_lb = self.sbss(distance_name, n_classes)

            np.savez(cache_file, folds=folds, fold_col_lb=fold_col_lb)

        # embaralhando os elementos das colunas (como se fossem random dos splits)
        np.random.seed(self.random_state)
        np.apply_along_axis(np.random.shuffle, 0, folds)

        # folds = np.array(folds_list).T
        for split in range(self.n_splits):
            train_splits = np.ones(self.n_splits).astype(np.bool)
            train_splits[split] = False

            test_index = folds[split]
            data["test"]["data"] = self.x[test_index]
            data["test"]["target"] = self.y[test_index]

            train_val_index = folds[train_splits, :].ravel()
            if validation_rate_from_train is None or validation_rate_from_train == 0:
                data["train"]["data"] = self.x[train_val_index]
                data["train"]["target"] = self.y[train_val_index]

                if self.dataset_name == "mnist_784":
                    data["train"]["data"] = self.x
                    data["train"]["target"] = self.y
                    data["test"]["data"] = self.fixed_x_test
                    data["test"]["target"] = self.fixed_y_test
            else:
                val = []
                for k in range(n_classes):
                    idx_k = self.y[train_val_index].squeeze() == k
                    n_validations = int(idx_k.sum() * validation_rate_from_train)
                    # folds__lb_k = folds[np.ix_(train_splits, fold_col_lb == k)]
                    folds__lb_k = folds[train_splits][:, fold_col_lb == k]
                    nb_train_folds, nb_k_in_fold = folds__lb_k.shape
                    nb_used_entire_folds = int(n_validations / nb_k_in_fold)

                    # Since each fold (row) retains shuffled samples of the same label (col), getting the first K
                    # is not a problem
                    tmp_val = []
                    tmp_val.extend(folds__lb_k[:nb_used_entire_folds, :].ravel())

                    # If complete folds were not sufficient
                    missing_samples = n_validations - len(tmp_val)
                    tmp_val.extend(
                        folds__lb_k[nb_used_entire_folds, :missing_samples].ravel()
                    )

                    val.extend(tmp_val)

                val_index = np.array(val).ravel()
                train_index = np.array(
                    list(set(train_val_index.ravel()) - set(val_index))
                )

                data["train"]["data"] = self.x[train_index]
                data["train"]["target"] = self.y[train_index]
                data["val"] = {}
                data["val"]["data"] = self.x[val_index]
                data["val"]["target"] = self.y[val_index]

                if self.dataset_name == "mnist_784":
                    merged_index = np.hstack([train_index, test_index])
                    data["train"]["data"] = self.x[merged_index]
                    data["train"]["target"] = self.y[merged_index]
                    data["val"]["data"] = self.fixed_x_test
                    data["val"]["target"] = self.fixed_y_test
                    data["test"]["data"] = self.fixed_x_test
                    data["test"]["target"] = self.fixed_y_test

            if self.one_hot_encode_output:
                set_one_hot_encode_output(data, self.one_hot_encoder)

            # Preprocessing
            if self.standardize:
                data = self.standardize_data(data)

            if self.normalize:
                data = self.normalize_data(data)

            if self.verbose:
                self.log.debug(scipy.stats.describe(data["train"]["data"]))
                self.log.debug(scipy.stats.describe(data["train"]["target"]))

            yield data

            # data['train']['data'] = x_tmp[train_index]
            # data['train']['target'] = self.one_hot_encoder.transform(y_tmp[train_index])
            # data['val'] = {}
            # data['val']['data'] = x_tmp[val_index]
            # data['val']['target'] = self.one_hot_encoder.transform(y_tmp[val_index])

    def get_splits_iter(self, validation_rate_from_train=None):
        data = {"train": {}, "test": {}}
        for i, (train_val_index, test_index) in enumerate(
            self.skf.split(self.x, self.y)
        ):
            data["test"]["data"] = self.x[test_index]
            data["test"]["target"] = self.y[test_index]

            x_tmp = self.x[train_val_index]
            y_tmp = self.y[train_val_index]

            # Using validation if needed
            if validation_rate_from_train is None or validation_rate_from_train == 0:
                data["train"]["data"] = x_tmp
                data["train"]["target"] = y_tmp
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=validation_rate_from_train,
                    random_state=self.random_state * i + i,
                )

                for train_index, val_index in sss.split(x_tmp, y_tmp):
                    data["train"]["data"] = x_tmp[train_index]
                    data["train"]["target"] = y_tmp[train_index]
                    data["val"] = {}
                    data["val"]["data"] = x_tmp[val_index]
                    data["val"]["target"] = y_tmp[val_index]

            if self.one_hot_encode_output:
                set_one_hot_encode_output(data, self.one_hot_encoder)

            # Preprocessing
            if self.standardize:
                data = self.standardize_data(data)

            if self.normalize:
                data = self.normalize_data(data)

            if self.verbose:
                self.log.debug(scipy.stats.describe(data["train"]["data"]))
                self.log.debug(scipy.stats.describe(data["train"]["target"]))

            yield data


def set_one_hot_encode_output(data, one_hot_encoder):
    data["train"]["target"] = one_hot_encoder.transform(data["train"]["target"])
    data["val"]["target"] = one_hot_encoder.transform(data["val"]["target"])
    data["test"]["target"] = one_hot_encoder.transform(data["test"]["target"])


def get_batches(x, y, batch_size):
    """get_batches

    :param x: ndarray
    :param y: ndarray
    :param batch_size: None if no batches

    :return: yield each batch as x, y
    """
    samples = range(y.shape[0])
    shuffle(samples)

    if batch_size is None:
        return x_tmp, y_tmp

    for offset in range(0, len(samples), batch_size):
        idxs = samples[offset : offset + batch_size]
        x_tmp = x[idxs]
        y_tmp = y[idxs]

        yield x_tmp, y_tmp


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors: torch.Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


# class MyDataloader(torch.utils.data.DataLoader)
