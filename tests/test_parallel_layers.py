#!/usr/bin/env python


from torch.functional import Tensor
from parallel_mlps.autoconstructive.model.parallel_mlp import (
    ParallelMLPs,
    build_model_ids,
)
import pytest
import torch
from torch import nn
from torch.optim import Adam

"""Tests for `parallel_mlps` package."""
import random
import torch
import os

import logging

logger = logging.getLogger()

# Reproducibility:
def reproducibility():
    torch.manual_seed(0)
    random.seed(0)
    # torch.set_deterministic(True)
    # torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


reproducibility()

N_SAMPLES = 5
N_FEATURES = 3
N_OUTPUTS = 2

MIN_NEURONS = 1
MAX_NEURONS = 3


@pytest.fixture()
def mocked_pmlps():
    pmlps = ParallelMLPs(
        in_features=3,
        out_features=2,
        hidden_neuron__model_id=[0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5],
        output__model_id=[0, 1, 2, 3, 4, 5],
        output__architecture_id=[0, 1, 2, 0, 1, 2],
        drop_samples=0,
        activations=[nn.Identity(), nn.Sigmoid()],
        input_perturbation_strategy=None,
        device="cpu",
    )
    with torch.no_grad():
        pmlps.hidden_layer.weight[:, :] = torch.Tensor(
            [
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
                [5, 5, 5],
                [6, 6, 6],
                [6, 6, 6],
                [6, 6, 6],
            ]
        )  # hidden vs. inputs, 12, 3
        pmlps.hidden_layer.bias[:] = torch.Tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        )
        pmlps.weight[:, :] = torch.Tensor(
            [
                [1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6],
                [1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 6, 6],
            ]
        )  # outputs, hidden 2, 12
        pmlps.bias[:, :] = torch.Tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
        )

    return pmlps


@pytest.mark.parametrize(
    "l,expected_regularization",
    [
        (1, [8, 29 / 2, 66 / 3, 35, 77 / 2, 135 / 3]),
        (
            2,
            [
                8,
                (40 + 4 + 9 + 8) / 2,
                (135 + 16 + 25 + 36 + 9 + 9) / 3,
                80 + 49 + 16 + 16,
                (250 + 64 + 81 + 25 + 25) / 2,
                (540 + 100 + 121 + 144 + 36 + 36) / 3,
            ],
        ),
    ],
)
def test_regularization(mocked_pmlps: ParallelMLPs, l, expected_regularization):
    reg = mocked_pmlps.get_regularization_term(gamma=1, l=l)
    expected_reg = torch.Tensor(expected_regularization)
    assert torch.equal(reg, expected_reg)
    print(reg)


@pytest.fixture
def X():
    return torch.rand(size=(N_SAMPLES, N_FEATURES))


@pytest.fixture
def Y():
    return torch.randint(low=0, high=2, size=(N_SAMPLES,))


@pytest.fixture
def activation_functions():
    return [nn.LeakyReLU(), nn.Sigmoid()]


@pytest.fixture
def parallel_mlp_object(activation_functions, X):
    hidden_neuron__model_id, outputs_ids, architecture_ids = build_model_ids(
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
        hidden_neuron__model_id=hidden_neuron__model_id,
        output__model_id=outputs_ids,
        output__architecture_id=architecture_ids,
        drop_samples=None,
        input_perturbation_strategy=None,
        activations=activation_functions,
        logger=logger,
        device="cpu",
    )


@pytest.mark.parametrize(
    "activation_functions,repetitions,min_neurons,max_neurons,step,expected,start,end,expected_output_ids,expected_architecture_ids,architecture_id,model_id,expected_activation,expected_num_neurons",
    [
        (
            [nn.ReLU(), nn.Sigmoid()],
            3,
            MIN_NEURONS,
            MAX_NEURONS,
            1,
            # fmt: off
            [0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 11, 11, 12, 13, 13, 14, 14, 14, 15, 16, 16, 17, 17, 17], #expected
            [0, 1, 3, 6, 7, 9, 12, 13, 15, 18, 19, 21, 24, 25, 27, 30,31, 33], #start
            [1, 3, 6, 7, 9, 12, 13, 15, 18, 19, 21, 24, 25, 27, 30,31, 33, 36], #end
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], #expected_output_ids
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5],
            # expected_architecture_ids
            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18, 19 ,20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, ]
            # [0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 11, 11, 12, 13, 13, 14, 14, 14, 15, 16, 16, 17, 17, 17, ],
            # [0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 11, 11, 12, 13, 13, 14, 14, 14, 15, 16, 16, 17, 17, 17, ],
            # fmt: on
            5,  # architecture_id
            11,  # model_id
            nn.Sigmoid(),  # expected_activation
            3,  # expected_num_neurons
        ),
        (
            [nn.ReLU(), nn.Sigmoid()],
            3,
            MIN_NEURONS,
            MAX_NEURONS,
            2,
            # fmt: off
            [0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 11, 11, 11], #expected
            [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21], #start
            [1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24], #end
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], #expected_output_ids
            [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3],
            # expected_architecture_ids
            # fmt: on
            2,  # architecture_id
            6,  # model_id
            nn.Sigmoid(),  # expected_activation
            1,  # expected_num_neurons
        ),
        (
            [nn.ReLU(), nn.Sigmoid()],
            3,
            MAX_NEURONS,
            MAX_NEURONS,
            1,
            # fmt: off
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], #expected
            [0, 3, 6, 9, 12, 15], #start
            [3, 6, 9, 12,15, 18], #end
            [0, 1, 2, 3, 4, 5], #expected_output_ids
            [0, 0, 0, 1, 1, 1],
            # expected_architecture_ids
            # fmt: on
            1,  # architecture_id
            4,  # model_id
            nn.Sigmoid(),  # expected_activation
            3,  # expected_num_neurons
        ),
        (
            [nn.ReLU(), nn.Sigmoid()],
            3,
            2,
            4,
            2,
            # fmt: off
            [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11], #expected
            [0, 2, 6, 8, 12, 14, 18, 20, 24, 26, 30, 32], #start
            [2, 6, 8, 12, 14, 18, 20, 24, 26, 30, 32, 36], #end
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], #expected_output_ids
            [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3],
            # expected_architecture_ids
            # fmt: on
            0,  # architecture_id
            4,  # model_id
            nn.ReLU(),  # expected_activation
            2,  # expected_num_neurons
        ),
        (
            [nn.ReLU(), nn.Tanh(), nn.Sigmoid()],
            2,
            2,
            4,
            2,
            # fmt: off
            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30, 31, 32, 33, 34, 35, ]
            [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11], #expected
            [0, 2, 6, 8, 12, 14, 18, 20, 24, 26, 30, 32], #start
            [2, 6, 8, 12, 14, 18, 20, 24, 26, 30, 32, 36], #end
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], #expected_output_ids
            [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5],
            # expected_architecture_ids
            # fmt: on
            5,  # architecture_id
            11,  # model_id
            nn.Sigmoid(),  # expected_activation
            4,  # expected_num_neurons
        ),
    ],
)
def test_build_model_ids(
    activation_functions,
    repetitions,
    min_neurons,
    max_neurons,
    step,
    expected,
    start,
    end,
    expected_output_ids,
    expected_architecture_ids,
    architecture_id,
    model_id,
    expected_activation,
    expected_num_neurons,
):
    hidden_layer_ids, output_ids, architecture_ids = build_model_ids(
        repetitions=repetitions,
        activation_functions=activation_functions,
        min_neurons=min_neurons,
        max_neurons=max_neurons,
        step=step,
    )

    assert expected == hidden_layer_ids
    if start is not None:
        pmlps = ParallelMLPs(
            in_features=N_FEATURES,
            out_features=N_OUTPUTS,
            hidden_neuron__model_id=hidden_layer_ids,
            output__model_id=output_ids,
            output__architecture_id=architecture_ids,
            drop_samples=False,
            input_perturbation_strategy="sqrt",
            activations=activation_functions,
            logger=logger,
            device="cpu",
        )
        assert (
            expected_num_neurons
            == pmlps.get_num_hidden_neurons_from_architecture_id(architecture_id)
        )
        assert expected_num_neurons == pmlps.get_num_hidden_neurons_from_model_id(
            model_id
        )
        assert type(expected_activation) == type(
            pmlps.get_activation_from_model_id(model_id)
        )
        assert model_id in pmlps.get_model_ids_from_architecture_id(architecture_id)
        assert torch.all(pmlps.model_id__start_idx.cpu() == torch.tensor(start))
        assert torch.all(pmlps.model_id__end_idx.cpu() == torch.tensor(end))
        assert torch.all(pmlps.model_id__end_idx.cpu() == torch.tensor(end))
        assert torch.all(
            pmlps.output__model_id.cpu() == torch.tensor(expected_output_ids)
        )
        assert torch.all(
            pmlps.output__architecture_id.cpu()
            == torch.tensor(expected_architecture_ids)
        )
    print(hidden_layer_ids)


@pytest.mark.parametrize(
    "repetitions,activations,min_neurons,max_neurons,step_neurons",
    [(3, [nn.LeakyReLU(), nn.Sigmoid()], 1, 4, 1)],
)
def test_architecture_ids(
    repetitions, activations, min_neurons, max_neurons, step_neurons
):
    (
        hidden_neuron__model_id,
        output__model_id,
        output__architecture_id,
    ) = build_model_ids(
        repetitions,
        activations,
        min_neurons,
        max_neurons,
        step_neurons,
    )
    pmlps = ParallelMLPs(
        3,
        2,
        hidden_neuron__model_id,
        output__model_id,
        output__architecture_id,
        None,
        None,
        activations,
        device="cpu",
    )
    pmlps.output__architecture_id
    print("a")


def test_fail_build_model_ids():
    with pytest.raises(ValueError, match=r".*only unique values.*"):
        build_model_ids(
            repetitions=2,
            activation_functions=[nn.ReLU(), nn.Sigmoid(), nn.Sigmoid()],
            min_neurons=MIN_NEURONS,
            max_neurons=MAX_NEURONS,
            step=1,
        )

    with pytest.raises(ValueError, match=r".*nn.Identity().*"):
        build_model_ids(
            repetitions=2,
            activation_functions=[],
            min_neurons=MIN_NEURONS,
            max_neurons=MAX_NEURONS,
            step=1,
        )


def test_parallel_single_mlps_forward(parallel_mlp_object: ParallelMLPs, X: Tensor):
    output = parallel_mlp_object(X)
    for i in parallel_mlp_object.unique_model_ids:
        mlp = parallel_mlp_object.extract_mlps([i])[0]
        output_mlp = mlp(X)
        assert torch.allclose(output[:, i, :], output_mlp)


def test_trainings(X, Y, parallel_mlp_object):
    reproducibility()
    lr = 1
    atol = 1e-8
    rtol = 0.99
    parallel_optimizer = Adam(params=parallel_mlp_object.parameters(), lr=lr)

    single_models = [
        parallel_mlp_object.extract_mlps([i])[0]
        for i in parallel_mlp_object.unique_model_ids
    ]
    single_optimizers = [
        Adam(params=model.parameters(), lr=lr) for model in single_models
    ]

    num_epochs = 100
    parallel_loss = nn.CrossEntropyLoss(reduction="none")
    sequential_loss = nn.CrossEntropyLoss()

    X = X.to(parallel_mlp_object.device)
    Y = Y.to(parallel_mlp_object.device)
    gradient = torch.ones(parallel_mlp_object.num_unique_models).to(X.device)

    for e in range(num_epochs):
        print(f"Epoch: {e}")
        parallel_optimizer.zero_grad()
        outputs = parallel_mlp_object(X)
        per_sample_candidate_losses = parallel_mlp_object.calculate_loss(
            parallel_loss, outputs, Y
        )
        candidate_losses = per_sample_candidate_losses.mean(0)
        candidate_losses.backward(gradient=gradient)
        parallel_optimizer.step()
        print(candidate_losses)
        print(parallel_mlp_object.hidden_layer.weight.mean())

        for i, (model, optimizer) in enumerate(zip(single_models, single_optimizers)):
            optimizer.zero_grad()
            single_outputs = model(X)
            loss = sequential_loss(single_outputs, Y)
            loss.backward()
            optimizer.step()

            # Asserts
            assert torch.allclose(candidate_losses[i], loss, atol=atol, rtol=rtol)

            m = parallel_mlp_object.extract_mlps([i])
            # assert torch.allclose(m[0].weight, model[0].weight, atol=atol, rtol=rtol)
            assert type(m[0]) == type(model)
            # assert torch.allclose(m[2].weight, model[2].weight, atol=atol, rtol=rtol)


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
