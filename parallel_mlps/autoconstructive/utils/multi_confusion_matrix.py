import torch
import pandas as pd
import numpy as np
from torch import Tensor


class MultiConfusionMatrix:
    """Confusion Matrix for Multiple Models. It will generage n_models confusion matrix of n_classes x n_classes.
    Rows = Ground Truth, Cols = Predictions
    """

    def __init__(
        self,
        n_models: int,
        n_classes: int,
        device: str = "cpu",
        model_ids: Tensor = None,
    ) -> None:
        self.n_models = n_models
        self.n_classes = n_classes
        self.model_ids = model_ids
        self.cm = torch.zeros(
            (self.n_models, self.n_classes, self.n_classes),
            device=device,
            dtype=torch.int,
        )
        self.current_cm = None
        self.n_models_arange = torch.arange(
            self.n_models, device=device, dtype=torch.long
        )
        self.one = torch.tensor([1], device=device, dtype=torch.int)
        self.device = device
        self._is_dirty = True

    def update(self, predictions: Tensor, targets: Tensor, mask: Tensor = None) -> None:
        """Updates the Confuson Matrix

        Args:
                predictions (Tensor): [batch_size, n_models]
                targets (Tensor): [batch_size]
                mask (Tensor): [batch_size, n_models]
        """
        self._is_dirty = True
        with torch.inference_mode():
            if predictions.ndim > 2:
                predictions = predictions.argmax(-1)

            if predictions.device != self.cm.device:
                predictions = predictions.to(self.cm.device)
            # cm[np.arange(self.n_models).repeat(batch_size), np.tile(np.arange(batch_size), n_models), targets.repeat(4).view(20), predictions.t().reshape(20)]=1
            batch_size = targets.shape[0]
            if self.current_cm is None or self.current_cm.shape[1] != batch_size:
                self.current_cm = (
                    torch.zeros(
                        self.n_models, batch_size, self.n_classes, self.n_classes
                    )
                    .int()
                    .to(self.cm.device)
                )

            self.current_cm *= 0
            self.current_cm[
                torch.arange(self.n_models).repeat_interleave(targets.shape[0]),
                torch.arange(targets.shape[0]).repeat(self.n_models),
                targets.repeat(self.n_models),
                predictions.t().reshape(-1),
            ] = 1

            if mask is not None:
                self.current_cm *= mask.t()[:, :, None, None]

            self.cm += self.current_cm.sum(1).int()
            # self.cm.index_put_(
            #     indices=(self.n_models_arange, targets[:, None], predictions),
            #     values=self.one,
            #     accumulate=True,
            # )

    @property
    def calculate_metrics(self):
        if self._is_dirty:
            metrics = {}
            self.total_samples = self.cm.sum(-1).sum(-1)[:, None]

            self.tp = self.cm.diagonal(dim1=-1, dim2=-2)
            self.fn = self.cm.sum(-1) - self.tp
            self.fp = self.cm.sum(-2) - self.tp
            self.tn = self.total_samples - self.fn - self.fp - self.tp

            self.total_samples = self.total_samples.squeeze()
            metrics["overall_acc"] = self.tp.sum(-1) / self.total_samples
            metrics["matthews_corrcoef"] = self._matthews_corrcoef()
            self._is_dirty = False
            self._calculate_metrics = metrics

        return self._calculate_metrics

    def to_dataframe(self, prefix=None):
        df = pd.DataFrame(self.calculate_metrics)
        if prefix is not None:
            df.columns = [f"{prefix}{c}" for c in df.columns]
        if self.model_ids is not None:
            df["model_id"] = self.model_ids.cpu().numpy()
            df = df.set_index("model_id")

        return df

    def _matthews_corrcoef(self):
        numerator = self.tp * self.tn - self.fp * self.fn
        denominator = torch.sqrt(
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )
        matt = numerator / denominator
        C = self.cm
        # C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        t_sum = C.sum(dim=-1)
        p_sum = C.sum(dim=-2)
        n_correct = self.tp.sum(-1)
        cov_ytyp = n_correct * self.total_samples - (t_sum * p_sum).sum(
            -1
        )  # torch.dot(t_sum, p_sum)
        cov_ypyp = self.total_samples ** 2 - (p_sum * p_sum).sum(
            -1
        )  # torch.dot(p_sum, p_sum)
        cov_ytyt = self.total_samples ** 2 - (t_sum * t_sum).sum(
            -1
        )  # torch.dot(t_sum, t_sum)

        matt = cov_ytyp / torch.sqrt(cov_ytyt * cov_ypyp)
        invalid_mask = (cov_ypyp * cov_ytyt) == 0

        matt[invalid_mask] = 0.000000001

        return matt