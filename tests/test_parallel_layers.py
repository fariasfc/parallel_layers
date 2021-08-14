#!/usr/bin/env python

"""Tests for `parallel_mlps` package."""

from parallel_mlps.parallel_mlp import ParallelMLPs, build_layer_ids
import pytest
import torch
from torch import nn

torch.manual_seed(0)


N_SAMPLES = 5
N_FEATURES = 3
N_OUTPUTS = 2

MIN_NEURONS = 1
MAX_NEURONS = 3


@pytest.fixture
def X():
    return torch.rand(size=(N_SAMPLES, N_FEATURES))


@pytest.fixture
def activation_functions():
    return [nn.ReLU(), nn.Sigmoid()]


def test_repeated_activation_functions():
    with pytest.raises(ValueError) as e:
        layers_ids = build_layer_ids(
            activation_functions=[nn.ReLU(), nn.Sigmoid(), nn.ReLU()],
            min_neurons=MIN_NEURONS,
            max_neurons=MAX_NEURONS,
            step=1,
        )

    assert "unique values" in str(e.value)


@pytest.fixture
def parallel_mlp_object(activation_functions, X):
    layer_ids = build_layer_ids(
        repetitions=3,
        activation_functions=activation_functions,
        min_neurons=MIN_NEURONS,
        max_neurons=MAX_NEURONS,
        step=1,
    )

    return ParallelMLPs(
        in_features=X.shape[1],
        out_features=N_OUTPUTS,
        bias=True,
        hidden_neuron__model_id=layer_ids,
        activations=activation_functions,
    )


@pytest.mark.parametrize(
    "activation_functions,repetitions,min_neurons,max_neurons,step,expected",
    [
        (
            [nn.ReLU(), nn.Sigmoid()],
            3,
            MIN_NEURONS,
            MAX_NEURONS,
            1,
            # fmt: off
            [0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 11, 11, 12, 13, 13, 14, 14, 14, 15, 16, 16, 17, 17, 17, ],
            # fmt: on
        ),
        (
            [nn.ReLU(), nn.Sigmoid()],
            3,
            MIN_NEURONS,
            MAX_NEURONS,
            2,
            # fmt: off
            [0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 11, 11, 11],
            # fmt: on
        ),
        (
            [nn.ReLU(), nn.Sigmoid()],
            3,
            MAX_NEURONS,
            MAX_NEURONS,
            1,
            # fmt: off
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
            # fmt: on
        ),
        (
            [nn.ReLU(), nn.Sigmoid()],
            3,
            2,
            4,
            2,
            # fmt: off
            [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11]
            # fmt: on
        ),
        (
            [nn.ReLU(), nn.Tanh(), nn.Sigmoid()],
            2,
            2,
            4,
            2,
            # fmt: off
            [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11]
            # fmt: on
        ),
    ],
)
def test_build_layer_ids(
    activation_functions, repetitions, min_neurons, max_neurons, step, expected
):
    layers_ids = build_layer_ids(
        repetitions=repetitions,
        activation_functions=activation_functions,
        min_neurons=min_neurons,
        max_neurons=max_neurons,
        step=step,
    )

    assert expected == layers_ids
    print(layers_ids)


def test_fail_build_layer_ids():
    with pytest.raises(ValueError, match=r".*only unique values.*"):
        build_layer_ids(
            repetitions=2,
            activation_functions=[nn.ReLU(), nn.Sigmoid(), nn.Sigmoid()],
            min_neurons=MIN_NEURONS,
            max_neurons=MAX_NEURONS,
            step=1,
        )

    with pytest.raises(ValueError, match=r".*nn.Identity().*"):
        build_layer_ids(
            repetitions=2,
            activation_functions=[],
            min_neurons=MIN_NEURONS,
            max_neurons=MAX_NEURONS,
            step=1,
        )


def test_parallel_mlp_forward(parallel_mlp_object, X):
    output = parallel_mlp_object(X)
    mlp = parallel_mlp_object.extract_mlp(2)
    output_mlp = mlp(X)
    assert torch.allclose(mlp, output_mlp)


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
