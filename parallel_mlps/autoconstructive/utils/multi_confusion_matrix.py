import torch
from torch import Tensor


class MultiConfusionMatrix:
    """Confusion Matrix for Multiple Models. It will generage n_models confusion matrix of n_classes x n_classes.
    Rows = Ground Truth, Cols = Predictions
    """

    def __init__(self, n_models: int, n_classes: int, device: str = "cpu") -> None:
        self.n_models = n_models
        self.n_classes = n_classes
        self.cm = torch.zeros(
            (self.n_models, self.n_classes, self.n_classes),
            device=device,
            dtype=torch.int,
        )
        self.n_models_arange = torch.arange(
            self.n_models, device=device, dtype=torch.long
        )
        self.one = torch.tensor([1], device=device, dtype=torch.int)
        self.device = device

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Updates the Confuson Matrix

        Args:
                predictions (Tensor): [batch_size, n_models]
                targets (Tensor): [batch_size]
        """
        with torch.inference_mode():
            if predictions.ndim > 2:
                predictions = predictions.argmax(-1)

            if predictions.device != self.cm.device:
                predictions = predictions.to(self.cm.device)

            self.cm.index_put_(
                indices=(self.n_models_arange, targets[:, None], predictions),
                values=self.one,
                accumulate=True,
            )

    def calculate_metrics(self):
        metrics = {}
        total_samples = self.cm[0].sum()

        self.tp = self.cm.diagonal(dim1=-1, dim2=-2)
        self.fn = self.cm.sum(-1) - self.tp
        self.fp = self.cm.sum(-2) - self.tp
        self.tn = total_samples - self.fn - self.fp - self.tp
        metrics["overall_acc"] = self.tp.sum(-1) / total_samples

        return metrics
