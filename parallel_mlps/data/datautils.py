import math
import pickle
from ctypes import ArgumentError

import numpy as np
import sklearn
import torch
import tqdm
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from data import dataloader
from data.samplers import BoostingBatchSampler, FastTensorDataLoader
from utils import ddu_utils, utils


def min_indices_2d(array, k):
    return np.unravel_index(np.argpartition(array.flatten(), k)[:k], array.shape)


def max_indices_2d(array, k):
    c = np.isnan(array).sum()

    ixs = np.unravel_index(
        np.argpartition(array.flatten(), -k - c)[-k - c : -c], array.shape
    )
    # idx = np.argpartition(array.ravel() , -k-c)[-k-cjj:-c]

    # val = ixs.flat[idx]
    return ixs


def torch_pairwise_correlation(x):
    """x = ndarray(n_samples, n_features)"""

    tx = torch.from_numpy(x).half()
    if torch.cuda.is_available():
        tx = tx.cuda()
    tx = tx - (tx.mean(1)[:, None])
    num = (tx[:, None] * tx[None, :]).sum(2)
    tx = torch.pow(tx, 2).sum(1)
    tx = torch.sqrt(tx[:, None] * tx[None, :])

    return (1 - num / tx).cpu().numpy()


def convert_labels_to_one_hot(labels):
    """Convert a array of int labels to its one-hot representation

    Args:
        labels (array): Array with n_samples elements ranging from 0 to N-1
            classes.

    Returns:
        one_hot_labels (array): 2d array with shape (n_samples, N classes) with
        1.0 at the position of each label.
    """
    shape = (labels.size, labels.max() + 1)
    one_hot_labels = np.zeros(shape).astype(np.float)

    rows = np.arange(labels.size)
    one_hot_labels[rows, labels] = 1.0

    return one_hot_labels


def generate_similar_batches(batch_size, labels, outputs, shuffle=None, how=None):
    """
    Gera batches que samples intralabel sao os mais diferentes possiveis

    Args:
        batch_size: int
        labels: boolean array (one vs all labels)
        shuffle: bool
        how: greatest_dissimilarity or lowest_dissimilarity

    Yields:
        *slice of batch_size elements*
    """
    INVALID_NUMBER = np.nan
    if how is None:
        how = "closest"
    unique_labels, counter_labels = np.unique(labels, return_counts=True)
    n = len(labels)

    assert np.all(unique_labels == np.arange(len(unique_labels)))

    # Cada label ficara com o mesmo numero de samples, divididos de maneira igualitaria.
    samples_per_lb_per_batch = int(batch_size / len(unique_labels))
    idxs = np.argsort(counter_labels)[::-1]

    # Relacionando label com seus indices
    lb__indexes = {lb: np.where(labels == lb)[0] for lb in unique_labels}

    # Shuffle nos indices de cada label
    if shuffle:
        for l in unique_labels:
            np.random.shuffle(lb__indexes[l])

    # o[sample_i, None, classificador_k] - o[None, sample_i, classificador_k]
    # d = distancia[sample_i, sample_j, calassificador_k]
    # d = np.abs(outputs[:, None, :] - outputs[None, :, :])
    distances = {}
    original_dissimilarities = {}
    lb_mask = {
        lb: np.zeros((outputs.shape[1], outputs.shape[1])) for lb in unique_labels
    }
    available_ixs = {}
    lb_original_ixs = {}
    for lb in unique_labels:
        rows_lb = labels == lb
        outputs_lb = outputs[rows_lb, :]
        lb_original_ixs[lb] = np.where(rows_lb)[0]
        dists = np.abs(outputs_lb[:, None, :] - outputs_lb[None, :, :])
        distances[lb] = np.mean(dists, axis=2)  # * np.std(dists, axis=2)
        available_ixs[lb] = np.ones(dists.shape[0]).astype(np.bool)
        # Setando diagonal infinita
        distances[lb] = distances[lb] + np.where(
            np.eye(distances[lb].shape[0]) > 0, INVALID_NUMBER, 0
        )
        original_dissimilarities[lb] = np.copy(distances[lb])
        # print(np.round(distances[lb], 2))

    # distances = np.abs(outputs[:, None, :] - outputs[None, :, :])

    # dissimilarity[sample_i, sample_j]
    # mediana das distancias entre as saidas i e j dentre todos os classificadores
    # samples que tem outputs parecidos (menor dissimilaridade) devem ficar em batches diferentes, aumentando o tamanho do gradiente geral
    # dissimilarities = np.median(distances, axis=2) * np.std(distances, axis=2)

    # marcadores dos splits para cada label
    starts = np.zeros(len(unique_labels)).astype(np.int)
    steps = (
        starts + samples_per_lb_per_batch
    )  # com 3 labels: [samples_per_lb_batch, samples_per_lb_batch, samples_per_lb_batch]

    # Se a soma das quantidades de cada label dentro de um batch for menor que batchsize, caso onde o batchsize nao eh multiplo
    # da quantidade de samples por labels por batch
    if np.sum(steps) < batch_size:
        diff = batch_size - np.sum(steps)
        # distribuindo de maneira flat para os labels
        for i in range(diff):
            steps[idxs[i]] = steps[idxs[i]] + 1
        # steps[idxs[0]] = steps[idxs[0]] + diff

    steps.copy()

    used_counters = np.copy(counter_labels)
    for i in range(int(n // batch_size)):
        batch_ixs = []
        for lb in unique_labels:
            distances_lb = distances[lb]
            available_ixs_lb = available_ixs[lb]
            # dissimilarities_lb[similar_ixs_lbs, :] = INVALID_NUMBER
            # dissimilarities_lb[similar_ixs_lbs, :] = INVALID_NUMBER
            similar_ixs_lbs = []
            # Enquanto faltar samples deste label
            while len(similar_ixs_lbs) < steps[lb]:
                # Pegar par menos similar
                if len(similar_ixs_lbs) == 0:
                    if used_counters[lb] >= 2:
                        if how == "closest":
                            chosen_ixs = list(
                                np.unravel_index(
                                    np.nanargmin(distances_lb), distances_lb.shape
                                )
                            )
                        else:
                            chosen_ixs = list(
                                np.unravel_index(
                                    np.nanargmax(distances_lb), distances_lb.shape
                                )
                            )
                    else:
                        chosen_ixs = np.where(available_ixs_lb)[0].tolist()
                else:
                    agg_distances_lb = np.nanmean(
                        original_dissimilarities[lb][similar_ixs_lbs, :], axis=0
                    )
                    agg_distances_lb[~available_ixs_lb] = INVALID_NUMBER

                    if how == "closest":
                        # Pegando indice mais similar a mediana dos que ja foram selecinoados
                        # chosen_ixs = [np.argpartition(agg_distances_lb, 0)[0]]
                        chosen_ixs = [np.nanargmin(agg_distances_lb)]
                    else:
                        # chosen_ixs = [np.argpartition(agg_distances_lb, -1)[-1:]]
                        chosen_ixs = [np.nanargmax(agg_distances_lb)]

                chosen_ixs = list(set(chosen_ixs).difference(set(similar_ixs_lbs)))

                if len(chosen_ixs) > 0:
                    used_counters[lb] -= len(chosen_ixs)
                    # Impossibilitando de pegar as mesmas coords
                    available_ixs_lb[chosen_ixs] = False
                    distances_lb[:, chosen_ixs] = INVALID_NUMBER
                    distances_lb[chosen_ixs, :] = INVALID_NUMBER

                    similar_ixs_lbs.extend(chosen_ixs)

                # used_counters[lb] -= 1
                # Se ja usei todos os labels, reseto a matriz de dissimilaridade
                if used_counters[lb] == 0:
                    # if np.all(distances_lb == np.nan):
                    distances[lb] = np.copy(original_dissimilarities[lb])
                    used_counters[lb] = counter_labels[lb]
                    available_ixs_lb[:] = True
                    # Se ainda nao terminou de pegar esse batch
                    if len(similar_ixs_lbs) < steps[lb]:
                        available_ixs_lb[similar_ixs_lbs] = False
            batch_ixs.extend(lb_original_ixs[lb][similar_ixs_lbs])
            # batch_ixs.extend(similar_ixs)

        yield np.array(batch_ixs)


#  try:
#      import cupy as cp
#      def generate_similar_batches_gpu(batch_size, labels, outputs, shuffle=None, how=None):
#          """
#          Gera batches que samples intralabel sao os mais diferentes possiveis
#
#          Args:
#              batch_size: int
#              labels: boolean array (one vs all labels)
#              shuffle: bool
#              how: greatest_dissimilarity or lowest_dissimilarity
#
#          Yields:
#              *slice of batch_size elements*
#          """
#
#          if how is None:
#              how = 'closest'
#
#          if how == 'closest':
#              INVALID_NUMBER = np.inf
#          else:
#              INVALID_NUMBER = -np.inf
#
#          outputs = cp.array(outputs)
#          unique_labels, counter_labels = np.unique(labels, return_counts=True)
#          n = len(labels)
#
#          assert np.all(unique_labels == np.arange(len(unique_labels)))
#
#          # Cada label ficara com o mesmo numero de samples, divididos de maneira igualitaria.
#          samples_per_lb_per_batch = int(batch_size/len(unique_labels))
#          idxs = np.argsort(counter_labels)[::-1]
#
#          # Relacionando label com seus indices
#          lb__indexes = {lb: np.where(labels == lb)[0] for lb in unique_labels}
#
#          # Shuffle nos indices de cada label
#          if shuffle:
#              for l in unique_labels:
#                  np.random.shuffle(lb__indexes[l])
#
#
#          # o[sample_i, None, classificador_k] - o[None, sample_i, classificador_k]
#          #d = distancia[sample_i, sample_j, calassificador_k]
#          # d = np.abs(outputs[:, None, :] - outputs[None, :, :])
#          distances = {}
#          original_dissimilarities = {}
#          available_ixs = {}
#          lb_original_ixs = {}
#          for lb in unique_labels:
#              rows_lb = labels == lb
#              outputs_lb = outputs[rows_lb, :]
#              lb_original_ixs[lb] =  np.where(rows_lb)[0]
#              dists = cp.abs(outputs_lb[:, None, :] - outputs_lb[None, :, :])
#              distances[lb] = cp.mean(dists, axis=2)# * np.std(dists, axis=2)
#              available_ixs[lb] = cp.ones(dists.shape[0]).astype(np.bool)
#              # Setando diagonal infinita
#              distances[lb] = distances[lb] + cp.where(cp.eye(distances[lb].shape[0]) > 0, INVALID_NUMBER,0)
#              original_dissimilarities[lb] = cp.copy(distances[lb])
#              # print(np.round(distances[lb], 2))
#
#          # distances = np.abs(outputs[:, None, :] - outputs[None, :, :])
#
#          # dissimilarity[sample_i, sample_j]
#          # mediana das distancias entre as saidas i e j dentre todos os classificadores
#          # samples que tem outputs parecidos (menor dissimilaridade) devem ficar em batches diferentes, aumentando o tamanho do gradiente geral
#          # dissimilarities = np.median(distances, axis=2) * np.std(distances, axis=2)
#
#          k = samples_per_lb_per_batch
#
#          # marcadores dos splits para cada label
#          starts = cp.zeros(len(unique_labels)).astype(np.int)
#          steps = starts + samples_per_lb_per_batch  # com 3 labels: [samples_per_lb_batch, samples_per_lb_batch, samples_per_lb_batch]
#
#          # Se a soma das quantidades de cada label dentro de um batch for menor que batchsize, caso onde o batchsize nao eh multiplo
#          # da quantidade de samples por labels por batch
#          if cp.sum(steps) < batch_size:
#              diff = batch_size - cp.sum(steps)
#              # distribuindo de maneira flat para os labels
#              for i in range(diff):
#                  steps[idxs[i]] = steps[idxs[i]] + 1
#              # steps[idxs[0]] = steps[idxs[0]] + diff
#
#          ends = steps.copy()
#
#          used_counters = np.copy(counter_labels)
#          for i in range(int(n // batch_size)):
#              batch_ixs = []
#              similar_ixs = []
#              for lb in unique_labels:
#                  distances_lb = distances[lb]
#                  available_ixs_lb = available_ixs[lb]
#                  # dissimilarities_lb[similar_ixs_lbs, :] = INVALID_NUMBER
#                  # dissimilarities_lb[similar_ixs_lbs, :] = INVALID_NUMBER
#                  similar_ixs_lbs = []
#                  # Enquanto faltar samples deste label
#                  while len(similar_ixs_lbs) < steps[lb]:
#                      # Pegar par menos similar
#                      if len(similar_ixs_lbs) == 0:
#                          if used_counters[lb] >= 2:
#                              if how == 'closest':
#                                  chosen_ixs = [c.item() for c in cp.unravel_index(cp.nanargmin(distances_lb), distances_lb.shape)]
#                              else:
#                                  chosen_ixs = [c.item() for c in cp.unravel_index(cp.nanargmax(distances_lb), distances_lb.shape)]
#                          else:
#                              chosen_ixs = cp.where(available_ixs_lb)[0].tolist()
#                      else:
#                          agg_distances_lb = cp.nanmean(original_dissimilarities[lb][similar_ixs_lbs, :], axis=0)
#                          agg_distances_lb[~available_ixs_lb] = INVALID_NUMBER
#
#                          if how == 'closest':
#                              # Pegando indice mais similar a mediana dos que ja foram selecinoados
#                              # chosen_ixs = [np.argpartition(agg_distances_lb, 0)[0]]
#                              chosen_ixs = [cp.nanargmin(agg_distances_lb).item()]
#                          else:
#                              # chosen_ixs = [np.argpartition(agg_distances_lb, -1)[-1:]]
#                              chosen_ixs = [cp.nanargmax(agg_distances_lb).item()]
#
#                      chosen_ixs = list(set(chosen_ixs).difference(set(similar_ixs_lbs)))
#
#                      if len(chosen_ixs) > 0:
#                          used_counters[lb] -= len(chosen_ixs)
#                          # Impossibilitando de pegar as mesmas coords
#                          available_ixs_lb[chosen_ixs] = False
#                          distances_lb[:, chosen_ixs] = INVALID_NUMBER
#                          distances_lb[chosen_ixs, :] = INVALID_NUMBER
#
#                          similar_ixs_lbs.extend(chosen_ixs)
#
#                      # used_counters[lb] -= 1
#                      # Se ja usei todos os labels, reseto a matriz de dissimilaridade
#                      if used_counters[lb] == 0:
#                      # if np.all(distances_lb == np.nan):
#                          distances[lb] = cp.copy(original_dissimilarities[lb])
#                          used_counters[lb] = counter_labels[lb]
#                          available_ixs_lb[:] = True
#                          # Se ainda nao terminou de pegar esse batch
#                          if len(similar_ixs_lbs) < steps[lb]:
#                              available_ixs_lb[similar_ixs_lbs] = False
#                  batch_ixs.extend(lb_original_ixs[lb][similar_ixs_lbs])
#                  # batch_ixs.extend(similar_ixs)
#
#              yield np.array(batch_ixs)
#  except Exception as e:
#      print('Could not load cupy')


# # nao chegamos no fim?
# lte = ends <= counter_labels

# indexes_slice = [lb__indexes[lb][starts[lb]:ends[lb]] for lb in unique_labels[lte]]
# indexes_slice = [item for sublist in indexes_slice for item in sublist]

# starts[lte] = ends[lte]
# ends[lte] += steps[lte]

# # Se o contador dos indices passou a quantidade maxima de samples existentes em algum label
# if np.any(lte==False):
#     for lb in unique_labels[~lte]:
#         end_tmp = counter_labels[lb]
#         missing_indexes_slice = lb__indexes[lb][starts[lb]:end_tmp]
#         diff = steps[lb] - len(missing_indexes_slice)

#         if diff > 0:
#             missing_indexes_slice = np.concatenate((missing_indexes_slice, [lb__indexes[lb][:diff]]), axis=None)

#         indexes_slice = np.concatenate((indexes_slice, missing_indexes_slice))

#         starts[lb] = diff
#         ends[lb] = diff + steps[lb]

# yield indexes_slice


def balanced_batch_generator(x, y, batch_size):
    def dataset_with_indices(cls):
        """
        Modifies the given Dataset class to return a tuple data, target, index
        instead of just data, target.
        """

        def __getitem__(self, index):
            cls.__getitem__(self, index)
            return index

        return type(
            cls.__name__,
            (cls,),
            {
                "__getitem__": __getitem__,
            },
        )

    unique_lb, class_sample_count = np.unique(y, return_counts=True)
    assert np.all(unique_lb == np.arange(len(unique_lb)))
    ixs = np.argsort(unique_lb)
    class_sample_count = class_sample_count[ixs]
    weights = y.copy().astype(float)
    for lb, sample_count in zip(unique_lb, class_sample_count):
        weights[weights == lb] = 1 / sample_count

    # class_sample_count = [10, 1, 20, 3, 4] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    # weights = 1 / torch.Tensor(class_sample_count)
    # weights = 1 / class_sample_count
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights), replacement=True
    )
    # TensorDatasetIxs = dataset_with_indices(torch.utils.data.TensorDataset)
    # trainloader = DataLoader(TensorDatasetIxs(x), batch_size=batch_size, sampler=sampler)
    trainloader = DataLoader(
        np.arange(x.shape[0]), batch_size=batch_size, sampler=sampler
    )
    # trainloader = DataLoader(np.arange(x.shape[0]), batch_size=batch_size, shuffle=False)#sampler=sampler)
    # return trainloader
    ixs = [i for i in trainloader]
    batches = [(x[i], torch.atleast_1d(torch.from_numpy(np.array(y[i])))) for i in ixs]
    return batches


def generate_batches_equally_amounts_label(
    batch_size, labels, shuffle=None, min_batch_size=0
):
    """
    Generate batches in a way that it grants the same amount of examples in each batch.
    If a number of samples in a given label is low, we repeat the samples in a round-robin way.
    We present all the data of this low sample label and the next batch that would not have samples of this class
    will present already viewed classes in order to the network 'not forget' that this class should be classified...

    Generator to create slices containing batch_size elements, from 0 to n.
    The last slice may contain less than batch_size elements, when batch_size
    does not divide n. :param n: :type n: int :param batch_size: Number of
    element in each batch :type batch_size: int :param min_batch_size: Minimum
    batch size to produce. :type min_batch_size: int, default=0 :param labels:
    The labels from 0 to N-1. :type labels: 1d array int :param shuffle: :type
    shuffle: bool

    Examples:
        >>> from sklearn.utils import gen_batches
        >>> list(gen_batches(7, 3))
        [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
        >>> list(gen_batches(6, 3))
        [slice(0, 3, None), slice(3, 6, None)]
        >>> list(gen_batches(2, 3))
        [slice(0, 2, None)]
        >>> list(gen_batches(7, 3, min_batch_size=0))
        [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
        >>> list(gen_batches(7, 3, min_batch_size=2))
        [slice(0, 3, None), slice(3, 7, None)]

    Args:
        n:
        batch_size:
        labels:
        shuffle:
        min_batch_size:

    Yields:
        *slice of batch_size elements*
    """
    unique_labels, counter_labels = np.unique(labels, return_counts=True)
    n = len(labels)

    assert np.all(unique_labels == np.arange(len(unique_labels)))

    # Cada label ficara com o mesmo numero de samples, divididos de maneira igualitaria.
    samples_per_lb_per_batch = int(batch_size / len(unique_labels))
    idxs = np.argsort(counter_labels)[::-1]

    # Relacionando label com seus indices
    lb__indexes = {lb: np.where(labels == lb)[0] for lb in unique_labels}

    # Shuffle nos indices de cada label
    if shuffle:
        for l in unique_labels:
            np.random.shuffle(lb__indexes[l])

    # marcadores dos splits para cada label
    starts = np.zeros(len(unique_labels)).astype(np.int)
    steps = (
        np.zeros(len(unique_labels)).astype(np.int) + samples_per_lb_per_batch
    )  # com 3 labels: [samples_per_lb_batch, samples_per_lb_batch, samples_per_lb_batch]

    # Se a soma das quantidades de cada label dentro de um batch for menor que batchsize, caso onde o batchsize nao eh multiplo
    # da quantidade de samples por labels por batch
    if np.sum(steps) < batch_size:
        diff = batch_size - np.sum(steps)
        # distribuindo de maneira flat para os labels
        for i in range(diff):
            steps[idxs[i]] = steps[idxs[i]] + 1
        # steps[idxs[0]] = steps[idxs[0]] + diff

    ends = steps.copy()

    for _ in range(int(n // batch_size)):
        # nao chegamos no fim?
        lte = ends <= counter_labels

        indexes_slice = [
            lb__indexes[lb][starts[lb] : ends[lb]] for lb in unique_labels[lte]
        ]
        indexes_slice = [item for sublist in indexes_slice for item in sublist]

        starts[lte] = ends[lte]
        ends[lte] += steps[lte]

        # Se o contador dos indices passou a quantidade maxima de samples existentes em algum label
        if np.any(lte == False):
            for lb in unique_labels[~lte]:
                end_tmp = counter_labels[lb]
                missing_indexes_slice = lb__indexes[lb][starts[lb] : end_tmp]
                diff = steps[lb] - len(missing_indexes_slice)

                if diff > 0:
                    missing_indexes_slice = np.concatenate(
                        (missing_indexes_slice, [lb__indexes[lb][:diff]]), axis=None
                    )

                indexes_slice = np.concatenate((indexes_slice, missing_indexes_slice))

                starts[lb] = diff
                ends[lb] = diff + steps[lb]

        yield indexes_slice


def sublabels_to_root(sublabels_map_root, outputs):
    r = outputs.clone()
    for (sublabel, root_lb) in sublabels_map_root.items():
        # ixs = outputs == sublabel
        ixs = torch.where(outputs == sublabel)
        r[ixs] = root_lb
    return r


# def extract_sublabels_from_outputs(labels, para):
#     unique_root_labels, count_root_lbs = np.unique(labels, return_counts=True)
#     sublabels = labels.copy()
#     min_prob = 0.5
#     sublabels_map_root = {}
#     for lb in unique_root_labels:
#         created_sublabel = True
#         # USAR ESSES CLASSIFICADORES
#         classifiers = [utils.get_classifier(params).fit(x, labels) for _ in range(n_classifiers)]
#         outputs = [c.predict_proba(x) for c in classifiers]
#         # # outputs = np.array([o.argmax(1) for o in outputs]).T
#         # outputs = np.array([(o>=(0.9*o.max())).argmax(1) for o in outputs]).T

#         # Pegando apenas a probabilidade da classse 1 (ja que eh binario)
#         outputs = np.array(outputs)[:, :, lb].T
#         while created_sublabel:
#             created_sublabel = False


#             tmp_sublabels, lb_sublabels_map_root = sublabel(outputs, sublabels, lb, min_prob, min_perc_elems_sublabel, count_root_lbs[lb])

#             if len(lb_sublabels_map_root) > 0:
#                 created_sublabel = True
#                 sublabels = tmp_sublabels
#                 sublabels_map_root = {**sublabels_map_root, **lb_sublabels_map_root}

#     return sublabels, sublabels_map_root


def extract_sublabels(params, n_classifiers, x, labels, min_perc_elems_sublabel):
    unique_root_labels, count_root_lbs = np.unique(labels, return_counts=True)
    sublabels = labels.copy()
    min_prob = 0.5
    sublabels_map_root = {}
    for lb in unique_root_labels:
        created_sublabel = True
        # USAR ESSES CLASSIFICADORES
        classifiers = [
            utils.get_classifier(params).fit(x, labels) for _ in range(n_classifiers)
        ]
        outputs = [c.predict_proba(x) for c in classifiers]
        # # outputs = np.array([o.argmax(1) for o in outputs]).T
        # outputs = np.array([(o>=(0.9*o.max())).argmax(1) for o in outputs]).T

        # Pegando apenas a probabilidade da classse 1 (ja que eh binario)
        outputs = np.array(outputs)[:, :, lb].T
        while created_sublabel:
            created_sublabel = False

            tmp_sublabels, lb_sublabels_map_root = sublabel(
                outputs,
                sublabels,
                lb,
                params["score_name"],
                min_prob,
                min_perc_elems_sublabel,
                count_root_lbs[lb],
            )

            if len(lb_sublabels_map_root) > 0:
                created_sublabel = True
                sublabels = tmp_sublabels
                sublabels_map_root = {**sublabels_map_root, **lb_sublabels_map_root}

    return sublabels, sublabels_map_root


def extract_sublabels_from_outputs(
    outputs, labels, min_perc_elems_sublabel, score_name
):
    unique_root_labels, count_root_lbs = np.unique(labels, return_counts=True)
    sublabels = labels.copy()
    min_prob = 0.5
    sublabels_map_root = {}
    for lb in unique_root_labels:
        created_sublabel = True
        # Pegando apenas a probabilidade da classse 1 (ja que eh binario)
        outputs = np.array(outputs)[:, :, lb].T
        while created_sublabel:
            created_sublabel = False

            tmp_sublabels, lb_sublabels_map_root = sublabel(
                outputs,
                sublabels,
                lb,
                score_name,
                min_prob,
                min_perc_elems_sublabel,
                count_root_lbs[lb],
            )

            if len(lb_sublabels_map_root) > 0:
                created_sublabel = True
                sublabels = tmp_sublabels
                sublabels_map_root = {**sublabels_map_root, **lb_sublabels_map_root}

    return sublabels, sublabels_map_root


def sublabel(
    outputs, labels, lb, score_name, min_prob, min_perc_elems_sublabel, count_root_lb
):
    sublabels_mapping_root = {}
    sublabels = labels.copy()
    rows_lb = labels == lb

    if len(np.unique(outputs)) > len(np.unique(sublabels)):  # if logits
        # rows_lb_expanded = rows_lb[:, None].expand(len(rows_lb), len(outputs.shape[1]))
        rows_lb_expanded = np.broadcast_to(
            rows_lb[:, None], (len(rows_lb), outputs.shape[1])
        )
        scores, thresholds = ddu_utils.get_scores(outputs, rows_lb_expanded, score_name)
        thresholded_outputs = outputs > thresholds
    else:
        thresholded_outputs = outputs

    # outputs_lb = (thresholded_outputs == lb).astype(np.int)
    outputs_in_lb = outputs * rows_lb.astype(np.int)[:, None]
    count_lb = rows_lb.sum()
    masked_outputs_lb = thresholded_outputs * rows_lb.astype(np.int)[:, None]
    nb_samples, nb_models = masked_outputs_lb.shape

    p = np.zeros((nb_samples, nb_samples))
    # similarities = np.zeros((nb_samples, nb_samples))
    # for m in range(nb_models):
    #     for i in range(nb_samples):
    #         for j in range(nb_samples):
    #             # if j > i:
    #             if True:
    #                 o_i = outputs[i, :]
    #                 o_j = outputs[j, :]

    #                 # diffs = np.abs(o_i-o_j)
    #                 # i = qnts % de j?.
    #                 # Rationale: similaridade de acordo com a saida do modelo (como o modelo enxerga a proximidade entre os samples).
    #                 # Os modelos podem ter distribuicoes de ativacao diferentes, por ex, modelo 1 tem um range [-1. 1], o modelo 2 tem um range [0.1, 0.3].
    #                 # Embora o range seja menor, pode ser que o modelo 2 tenha uma separabilidade melhor.
    #                 #
    #                 # saidas parecidas => similarity = 100% => exemplos proximos (mesmo cluster)
    #                 # saidas distantes => similarity = 0% => exemplos muito diferentes
    #                 # Ex: modelo1: i=0.9, j = 0.3, modelo2: i=0.3, j = 0.2
    #                 # similaridade1 = 0.3/0.9=33%
    #                 # similaridade2= 0.2/0.3 = 66%
    #                 # similaridade = media(sim1, sim2...) = 49.5%
    #                 #
    #                 similarity = np.mean(np.minimum(o_i, o_j) / (np.maximum(o_i, o_j)  + 1e-10)) * rows_lb[i] * rows_lb[j]
    #                 similarities[i, j] = similarity
    #                 p[i, j] = similarity

    tmp_p = np.minimum(outputs_in_lb[:, None], outputs_in_lb[None, :]) / (
        np.maximum(outputs_in_lb[:, None], outputs_in_lb[None, :]) + 1e-10
    )
    dists = np.abs(outputs_in_lb[:, None] - outputs_in_lb[None, :])
    dists = (dists.max() - dists) / (dists.max() - dists.min())
    dists = dists * masked_outputs_lb
    dists = np.mean(dists, axis=2)

    # p[p == 0] = np.nan
    tmp_p = masked_outputs_lb[:, None] * masked_outputs_lb[None, :]
    tmp_p = np.mean(tmp_p, axis=2)
    # p = np.minimum(outputs_in_lb[:, None], outputs_in_lb[None, :])/(np.maximum(outputs_in_lb[:, None], outputs_in_lb[None, :])+1e-10)
    # p = np.mean(tmp_p, axis=2)
    p = tmp_p * dists
    # p[p == 0] = np.nan
    # p = np.nanmean(p, axis=2)
    # p[np.isnan(p)] = 0
    # p = np.mean(tmp_p*dists, axis=2)
    p = np.triu(
        p, k=1
    )  # Pegando apenas o triangulo superior e ignorando a diagnoal (nao faz sentido pegar o i e i novamente como par parecido.)
    # Cancelando indices de labels diferentes
    p[:, ~rows_lb] = 0
    p[~rows_lb, :] = 0

    # for i in range(masked_outputs_lb.shape[0]):
    #     for j in range(masked_outputs_lb.shape[0]):
    #         if j > i:
    #             o_i = masked_outputs_lb[i, :]
    #             o_j = masked_outputs_lb[j, :]
    #             # w_i = output[i, :]
    #             # w_j = output[j, :]

    #             o_and = o_i*o_j
    #             p[i, j] = o_and.mean()

    # print(f'p: \n{p}')
    keep_going = True
    sublabel_samples = []
    prob_labels = []
    i = j = None
    p_apriori = np.mean(p, axis=1)
    while keep_going:
        keep_going = False

        if len(sublabel_samples) == 0:
            i, j = np.unravel_index(np.argmax(p), p.shape)

            sublabel_samples.append(i)
            sublabel_samples.append(j)
            prob_labels.append(p[i, j])
            p[:, i] = 0
            p[:, j] = 0
            keep_going = True
        else:
            tmp_p = p[sublabel_samples, :]

            i, j = np.unravel_index(np.argmax(tmp_p), tmp_p.shape)
            i = sublabel_samples[i]
            prob = p[i, j]

            # # tmp_p = np.mean(tmp_p, axis=0)
            # # tmp_p = np.sum(tmp_p, axis=0)
            # tmp_p = np.nanprod(tmp_p, axis=0)
            # j = np.argmax(tmp_p)
            # prob = tmp_p[j]

            # print(prob)
            # if prob >= min_prob:
            # pr = prob/prob_labels[0]
            # pr = np.prod(prob_labels)*prob
            prob = prob * prob_labels[-1]
            pr = prob
            # pr = prob * prob_labels[-1]
            # print(f'{i}, {j}, prob/prob_labels[0]={prob}/{prob_labels[0]}={prob/prob_labels[0]}')
            if pr >= min_prob:
                prob_labels.append(prob)
                p[:, j] = 0
                sublabel_samples.append(j)
                keep_going = True
    # sublabels < quantidade de elementso desse label (pra nao apenas fazer um replace do label total)
    remaining_root_lb = count_root_lb - len(sublabel_samples)
    perc_remaining_root_lb = remaining_root_lb / count_root_lb
    nb_sublabel_samples = len(sublabel_samples)
    perc_sublabel_samples = nb_sublabel_samples / count_root_lb
    if (
        nb_sublabel_samples < count_lb
        and perc_sublabel_samples >= min_perc_elems_sublabel
        and perc_remaining_root_lb >= min_perc_elems_sublabel
    ):
        sublabel = np.max(sublabels) + 1
        sublabels[sublabel_samples] = sublabel
        sublabels_mapping_root[sublabel] = lb

    print(prob_labels[:10])
    return sublabels, sublabels_mapping_root


TPB = 12
bits = 32
MAX_FEATURES = 784


# TODO: descomentar... comentei pra rodar no macos
# @cuda.jit  # ("void(float{}[:, :], int{}[:], int{}[:], float{}[:], int{})".format(bits, bits, bits, bits, bits))
def correlation_distance_i(
    x, unpicked_samples, picked_samples, correlation_values, horizontal_stride
):
    x.shape[0]
    n = x.shape[1]
    nb_umpicked = len(unpicked_samples)
    nb_picked = len(picked_samples)
    i, j = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    cuda.gridDim.x
    cuda.gridDim.y
    bwy = cuda.blockDim.y
    cuda.blockIdx.y
    # s_chosen_ix = cuda.shared.array(shape=(TPB, MAX_FEATURES), dtype=np.float32)
    s_rows = cuda.shared.array(shape=(TPB, MAX_FEATURES), dtype=np.float32)
    to_sum = cuda.shared.array(shape=(TPB, TPB), dtype=np.float32)

    # if i >= correlation_values.shape[0] or j >= correlation_values.shape[1]:
    #     return

    start_chosen_ix = j * bwy
    end_chosen_ix = min(start_chosen_ix + bwy, nb_picked)
    end_chosen_ix - start_chosen_ix + 1

    # if i < nb_umpicked and start_chosen_ix < nb_picked:
    if i < nb_umpicked and j < nb_picked:
        # if i < correlation_values.shape[0] and j < correlation_values.shape[1]:
        # if j * bwy + ty > nb_picked:
        #     return
        # print(i)
        # print(j)
        # print(start_chosen_ix)
        # print(end_chosen_ix)
        # print('--------------')
        # cuda.syncthreads()
        row = unpicked_samples[i]
        for k in range(n):
            s_rows[tx, k] = x[row, k]

        col = picked_samples[j]
        # for k in range(n):
        #     s_chosen_ix[ty, k] = x[col, k]

        # cuda.syncthreads()

        picked_samples_offset = cuda.blockIdx.y * cuda.blockDim.y
        rg = cuda.blockDim.y
        if picked_samples_offset + rg >= nb_picked:
            rg = nb_picked - picked_samples_offset

        u = s_rows[tx, :n]

        # sum_similarity = 0
        # v = s_chosen_ix[ty, :n]
        # u = x[row, :]
        v = x[col, :]

        u_mean = 0
        v_mean = 0
        for k in range(n):
            u_mean += u[k]
            v_mean += v[k]

        u_mean = u_mean / n
        v_mean = v_mean / n

        num = 0

        du = 0
        dv = 0
        for k in range(n):
            num += (u[k] - u_mean) * (v[k] - v_mean)
            du += math.pow(u[k] - u_mean, 2)
            dv += math.pow(v[k] - v_mean, 2)

        if row == col:
            to_sum[tx, ty] = 0
        else:
            to_sum[tx, ty] = 1 - num / (math.sqrt(du) * math.sqrt(dv))

        cuda.syncthreads()

        if ty == 0:
            accum = 0
            for k in range(rg):
                accum += to_sum[tx, k]

            cuda.syncthreads()

            correlation_values[i, cuda.blockIdx.y] = accum
        # correlation_values[i, j] = accum
        # correlation_values[i, j] = accum
        # sum_similarity += similarity_corr
        # if similarity_corr < best_similarity:
        #     best_similarity = similarity_corr
        #     smallest_i = row
        #     smallest_j = col

        ###################
        # sum_similarity = 0
        # counter = -1
        # for c_ix in range(start_chosen_ix, end_chosen_ix):#chosen_ixs:
        #     col = picked_samples[c_ix]
        # # for c_ix in range(block_len_chosen_ix):#chosen_ixs:
        #     # col = chosen_ixs[start_chosen_ix + c_ix]
        #     counter += 1
        #     if col != row:
        #         v = s_chosen_ix[counter, :n]
        #         # v = x[col, :]
        #         ##### v = x[col, :]

        #         u_mean = 0
        #         v_mean = 0
        #         for k in range(n):
        #             u_mean += u[k]
        #             v_mean += v[k]

        #         u_mean = u_mean/n
        #         v_mean = v_mean/n

        #         num = 0

        #         du = 0
        #         dv = 0
        #         for k in range(n):
        #             num += (u[k]-u_mean) * (v[k]-v_mean)
        #             du += math.pow(u[k]-u_mean, 2)
        #             dv += math.pow(v[k]-v_mean, 2)

        #         similarity_corr = 1 - num/(math.sqrt(du)*math.sqrt(dv))
        #         sum_similarity += similarity_corr
        #         if similarity_corr < best_similarity:
        #             best_similarity = similarity_corr
        #             smallest_i = row
        #             smallest_j = col

        # correlation_values[i, j] = sum_similarity


def apply_sublabels_repeatedly(
    outputs_train,
    labels_train,
    outputs_val,
    labels_val,
    lb,
    min_prob,
    min_perc_elems_sublabel,
):
    nb_train_samples = outputs_train.shape[0]

    outputs = torch.cat((outputs_train, outputs_val), axis=0)
    labels = np.concatenate((labels_train, labels_val))
    sublabels = labels.copy()
    sublabels_train = labels_train
    sublabels_val = labels_val

    sublabels_map_root = {}
    created_sublabel = True

    count_root_lb = (labels == lb).sum()

    while created_sublabel:
        created_sublabel = False

        tmp_sublabels, lb_sublabels_map_root = apply_sublabel_output_probability(
            outputs, sublabels, lb, count_root_lb, min_prob, min_perc_elems_sublabel
        )

        if len(lb_sublabels_map_root) > 0:
            created_sublabel = True
            sublabels = tmp_sublabels
            sublabels_map_root = {**sublabels_map_root, **lb_sublabels_map_root}
            sublabels_train = sublabels[:nb_train_samples]
            sublabels_val = sublabels[nb_train_samples:]

    return sublabels_train, sublabels_val, sublabels_map_root


def apply_sublabel_output_probability(
    outputs, labels, lb, count_root_lb, min_prob, min_perc_elems_sublabel
):
    INVALID_NUMBER = -np.inf

    sublabels = labels.copy()
    sublabels_map_root = {}

    similarity = {}

    rows_lb = labels == lb
    outputs_lb = outputs[rows_lb, :]
    outputs_lb = torch.from_numpy(
        utils.min_max_matrix(outputs_lb.cpu().numpy(), axis=0)
    ).to(outputs.device)
    original_ixs = np.where(rows_lb)[0]

    # dists [sample_i, sample_, classifier_k]
    d = torch.abs(outputs_lb[:, None, :] - outputs_lb[None, :, :])
    # Media das ativacoes dos classificadores para cada sample
    mean_o = torch.mean(outputs_lb, axis=1).cpu().numpy()
    mean_d = torch.mean(d, axis=2).cpu().numpy()  # * np.std(dists, axis=2)
    del d
    torch.cuda.empty_cache()
    sim = 1 - utils.min_max_matrix(mean_d)

    # Setando diagonal -infinito (simulando maxima dissimilaridade)
    similarity = sim + np.where(np.eye(sim.shape[0]) > 0, INVALID_NUMBER, 0)

    # Quanto maior a ativacao e a similaridade, mais provavel os samples de serem parecidos
    wp = similarity * mean_o

    samples = []
    most_similar_samples = np.unravel_index(np.argmax(wp), wp.shape)
    samples = list(most_similar_samples)
    previous_prob = wp[most_similar_samples]
    utils.set_row_cols_(wp, INVALID_NUMBER, cols=samples)
    lowest_prob = np.inf

    while True:
        tmp_wp = wp[samples]
        most_similar_samples = np.unravel_index(np.argmax(tmp_wp), tmp_wp.shape)
        current_prob = tmp_wp[most_similar_samples]

        prob = current_prob / previous_prob

        if prob < min_prob:
            break

        lowest_prob = prob if prob < lowest_prob else lowest_prob
        previous_prob = current_prob
        s = most_similar_samples[1]
        samples.append(s)
        utils.set_row_cols_(wp, INVALID_NUMBER, cols=[s])

    # Se o sublabel >= minimo % do root label e o que restara de root label permanece maior que o minimo de % do root label
    if (
        len(samples) / count_root_lb >= min_perc_elems_sublabel
        and (count_root_lb - len(samples)) / count_root_lb >= min_perc_elems_sublabel
    ):
        # Trazendo para os indices originais:
        samples = [original_ixs[i] for i in samples]
        new_sublabel = np.max(sublabels) + 1
        sublabels[samples] = new_sublabel
        sublabels_map_root[new_sublabel] = lb

    return sublabels, sublabels_map_root


def gpu_correlation(x, unpicked_lb_samples, picked_lb_samples):
    # unpicked_lb_samples = np.where(rows_lb)[0]
    cuda.get_current_device()

    stream = cuda.stream()
    # blk = int(device.WARP_SIZE/2)
    # TPB = int(device.WARP_SIZE)
    len_picked = len(picked_lb_samples)
    len_unpicked = len(unpicked_lb_samples)
    # threads_per_block = (min(warp, len_unpicked), min(warp, int(len_picked/blk + 1)))
    threads_per_block = (TPB, TPB)
    x_grid = int(len_unpicked / threads_per_block[0] + 1)
    y_grid = int(len_picked / threads_per_block[1] + 1)
    blocks_per_grid = (x_grid, y_grid)

    correlation_values = cuda.to_device(
        np.zeros((len(unpicked_lb_samples), y_grid)).astype(np.float32)
    )
    # correlation_values = cuda.device_array((len(unpicked_lb_samples), y_grid), dtype=np.float32)

    correlation_distance_i[blocks_per_grid, threads_per_block](
        x,
        np.array(unpicked_lb_samples),
        np.array(picked_lb_samples),
        correlation_values,
        threads_per_block[1],
    )

    correlations = correlation_values.copy_to_host(stream=stream).astype(np.float32)
    correlations = np.sum(correlations, axis=1)

    best_correlation = np.argmin(correlations)
    best_correlation = np.argpartition(correlations, 0)[0]
    cuda_ix = unpicked_lb_samples[best_correlation]
    return cuda_ix


def gpu_sbss(x, labels, n_folds, similarity_name="correlation"):
    assert x.dtype == np.float32
    stream = cuda.stream()
    lb_samples = {}
    # changing the objects reference
    x = x.copy()
    labels = labels.copy()
    unique_labels = np.unique(labels)

    folds = [[] for _ in range(n_folds)]
    fold_labels = []

    for lb in tqdm.tqdm(unique_labels, desc="SBSS"):
        lb_samples[lb] = []
        rows_lb = labels == lb
        unpicked_lb_samples = np.where(rows_lb)[0].tolist()
        x_lb = cuda.to_device(np.asarray(x, dtype=x.dtype), stream=stream)
        while len(unpicked_lb_samples) >= n_folds:
            picked_lb_samples = []
            for split in range(n_folds):
                print(split)
                if split == 0:
                    most_similar = gpu_correlation(
                        x_lb, unpicked_lb_samples, unpicked_lb_samples
                    )
                else:
                    most_similar = gpu_correlation(
                        x_lb, unpicked_lb_samples, picked_lb_samples
                    )

                picked_lb_samples.append(most_similar)
                unpicked_lb_samples.remove(most_similar)
            lb_samples[lb].append(most_similar)

            shuffle(picked_lb_samples)

            for split in range(n_folds):
                folds[split].append(picked_lb_samples[split])
                fold_labels.append(lb)

    with open("/tmp/lb_samples_gpu.pk", "wb") as f:
        pickle.dump(lb_samples, f)

    return np.array(folds), np.array(fold_labels)


def gpu_similar_sort(x, labels, similarity_name="correlation"):
    np_type = np.float32
    device = cuda.get_current_device()

    stream = cuda.stream()
    mat2 = cuda.to_device(np.asarray(x, dtype=np_type), stream=stream)
    # ignore_ixs = cuda.device_array(rows-1, dtype=np.int32)
    # np_ignore_ixs = np.zeros(rows-1).astype(np.int) - 1
    np_ignore_ixs = set()
    lb_samples = {}
    for lb in np.unique(labels):
        lb_samples[lb] = []
        ignore_ixs = cuda.to_device(
            np.array(list(np_ignore_ixs)).astype(np.int32), stream=stream
        )
        # rows = mat.shape[0]
        rows_lb = labels == lb
        chosen_ixs = []
        while rows_lb.sum() > 0:
            rows = np.where(rows_lb)[0]
            block_dim = device.WARP_SIZE
            grid_dim = int(len(rows) / block_dim + 1)

            correlation_values = cuda.device_array(len(rows), dtype=np.float32)

            if similarity_name == "correlation":
                correlation_distance_i[grid_dim, block_dim](
                    mat2, rows, np.array(chosen_ixs), correlation_values
                )
            else:
                raise ArgumentError("Similarity name unrecognized")

            correlations = correlation_values.copy_to_host(stream=stream)

            best_correlation = np.argmin(correlations)
            cuda_ix = rows[best_correlation]

            chosen_ixs.append(cuda_ix)
            rows_lb[cuda_ix] = False
            lb_samples[lb].append(cuda_ix)

    return lb_samples


def get_normalization_coefficients(outputs, range=(0.1, 0.9)):
    """find scale and min values to normalize the outputs.
    Args:
        outputs [n_samples, n_models]
        scale [n_models]
        min [n_models]
    """
    if type(outputs) == torch.Tensor:
        outputs = outputs.cpu().detach().numpy()
    scaling = sklearn.preprocessing.MinMaxScaler(range)
    scaling.fit(outputs)
    scale_ = scaling.scale_
    min_ = scaling.min_
    scale_ = 1 / (outputs.max(0) - outputs.min(0))

    # In case there are infs...
    scale_ = np.where(np.isfinite(scale_), scale_, 1)

    min_ = outputs.min(0)

    return scale_, min_


def get_dataloader(
    model, batch_generator, t_x_train, t_target_train, labels_train, batch_size
):
    # Garantindo que os datasets fiquem intactos...
    t_x_train = t_x_train.clone()
    t_target_train = t_target_train.clone()
    labels_train = labels_train.clone().cpu().numpy().astype(np.long)  # numpy

    if batch_generator == "balanced_batch_generator":
        batch_indexes = balanced_batch_generator(t_x_train, labels_train, batch_size)
    elif batch_generator == "generate_batches_equally_amounts_label":
        batch_indexes = generate_batches_equally_amounts_label(
            batch_size, labels_train, shuffle=True
        )
    elif batch_generator == "generate_similar_batches":
        with torch.no_grad():
            outputs = model.forward(t_x_train)
            outputs = utils.candidate_output_to_2d(outputs)
        batch_indexes = generate_similar_batches(
            batch_size, labels_train, outputs.cpu().numpy(), how="farthest"
        )
    elif batch_generator == "pytorch":
        # my_dataset = dataloader.MyDataset(t_x_train, t_target_train)
        my_dataset = TensorDataset(t_x_train, t_target_train)
        if model.cfg.data.all_data_to_cuda:
            num_workers = 0
        else:
            num_workers = 6

        # sampler = BalancedBatchSampler(my_dataset, labels_train)
        batch_indexes = torch.utils.data.DataLoader(
            my_dataset,
            batch_size,
            drop_last=False,
            num_workers=num_workers,
            shuffle=model.cfg.data.shuffle_dataloader,
            pin_memory=not model.cfg.data.all_data_to_cuda,
        )
    elif batch_generator == "pytorch_shuffle":
        batch_indexes = FastTensorDataLoader(
            t_x_train,
            t_target_train,
            batch_size=batch_size,
            shuffle=model.cfg.data.shuffle_dataloader,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
        )
    elif batch_generator == "pytorch_boosting":
        my_dataset = dataloader.MyDataset(t_x_train, t_target_train)
        with torch.no_grad():
            outputs = model.forward(t_x_train.to(model.device))
            outputs = utils.candidate_output_to_2d(outputs).cpu()

        sampler = BoostingBatchSampler(
            my_dataset,
            outputs,
            torch.from_numpy(labels_train),
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
        )
        batch_indexes = torch.utils.data.DataLoader(
            my_dataset,
            batch_size,
            drop_last=True,
            pin_memory=False,
            num_workers=0,
            sampler=sampler,
        )
    else:
        raise ValueError(f"batch_generator not recognized {batch_generator}")

    return batch_indexes
