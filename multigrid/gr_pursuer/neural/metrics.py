import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

metric_funcs = {
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0),
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro", zero_division=0), 
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro", zero_division=0), 
}

def compute_metrics(y_true, y_pred):
    metrics = {}
    for metric in metric_funcs.keys():
        metrics[metric] = metric_funcs[metric](y_true, y_pred)
    return metrics

class CumulativeMetrics():

    def __init__(self, labels, posfix=None):
        self.values = {
            "FP": None,
            "FN": None,
            "TP": None,
            "TN": None,
        }
        self.count = 0
        self.labels = labels
        self.prefix = posfix

    def update(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)

        input_metrics = {
            "FP": cm.sum(axis=0) - np.diag(cm),
            "FN": cm.sum(axis=1) - np.diag(cm),
            "TP": np.diag(cm),
        }   
        input_metrics["TN"] = cm.sum() - sum(input_metrics.values())
        # Update the values
        for k in self.values.keys():
            if self.values[k] is None:
                self.values[k] = input_metrics[k]
            else:
                self.values[k] = self.values[k] + input_metrics[k]


    def compute(self):
        metrics = {}
        # metrics.update(self.values)
        # Get Values
        TP = self.values["TP"].astype(float)
        TN = self.values["TN"].astype(float)
        FP = self.values["FP"].astype(float)
        FN = self.values["FN"].astype(float)

        metrics["precision"] = np.divide(TP,(TP+FP), out=np.zeros_like(TP), where=(TP+FP)!=0).mean() 
        metrics["recall"] = np.divide(TP,(TP+FN), out=np.zeros_like(TP), where=(TP+FP)!=0).mean() 
        metrics["f1"] = 2*(metrics["precision"]*metrics["recall"])/(metrics["precision"]+metrics["recall"]).mean()
        metrics["accuracy"] = ((TP+TN)/(TP+TN+FP+FN)).mean()

        if self.prefix is not None:
            metrics = {f"{k}_{self.prefix}": v for k, v in metrics.items()}

        return metrics