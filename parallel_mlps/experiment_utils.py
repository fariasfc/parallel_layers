import sklearn.preprocessing
import numpy as np
import pycm


def assess_model(logits, y_labels, metric_prefix):
    if not isinstance(logits, dict):
        logits = {"": logits}

    y = sklearn.preprocessing.OneHotEncoder().fit_transform(y_labels[:, None]).todense()

    metrics_list = []
    for model, tmp_logits in logits.items():
        if tmp_logits.ndim > 1:
            tmp_logits_labels = tmp_logits.argmax(1)

        cm = pycm.ConfusionMatrix(y_labels, tmp_logits_labels)
        metrics = {}
        metrics[f"{metric_prefix}_{model}_aucroc_micro"] = float(
            sklearn.metrics.roc_auc_score(y, tmp_logits, average="micro")
        )
        metrics[f"{metric_prefix}_{model}_aucroc_macro"] = float(
            sklearn.metrics.roc_auc_score(y, tmp_logits, average="macro")
        )
        metrics[f"{metric_prefix}_{model}_aucroc_weighted"] = float(
            sklearn.metrics.roc_auc_score(y, tmp_logits, average="weighted")
        )
        metrics[f"{metric_prefix}_{model}_aucroc_samples"] = float(
            sklearn.metrics.roc_auc_score(y, tmp_logits, average="samples")
        )
        metrics[f"{metric_prefix}_{model}_matthews_corrcoef"] = float(
            sklearn.metrics.matthews_corrcoef(y_labels, tmp_logits_labels)
        )
        metrics[f"{metric_prefix}_{model}_overall_acc"] = cm.Overall_ACC
        metrics[f"{metric_prefix}_{model}_f1_macro"] = cm.F1_Macro
        metrics[f"{metric_prefix}_{model}_f1_micro"] = cm.F1_Micro
        metrics[f"{metric_prefix}_{model}_confusion_matrix"] = str(cm.matrix).replace(
            " ", ""
        )
        # precision, recall, _ = sklearn.metrics.precision_recall_curve(testy, naive_probs)
        # auc_precision_recall = sklearn.metrics.auc(recall, precision)

        # print(f'{metric_prefix}  -  {model} - mcc: {metrics["matthews_corrcoef"]} aucroc_micro: {metrics["aucroc_micro"]} aucroc_macro: {metrics["aucroc_macro"]} ACC: {cm.Overall_ACC} F1_Macro: {cm.F1_Macro} F1_Micro: {cm.F1_Micro}')
        # print(f'{metric_prefix}  -  {model} - aucrocweighted: {cm.overall_stat["aucroc_weighted"]}  ACC: {cm.Overall_ACC} F1_Macro: {cm.F1_Macro} F1_Micro: {cm.F1_Micro}')

        metrics_list.append(metrics)

    return metrics_list