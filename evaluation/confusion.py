from sklearn.metrics import confusion_matrix
import numpy as np

def compute_confusion_matrix(true_labels, predictions, num_classes):
    return confusion_matrix(true_labels, predictions, labels=list(range(num_classes)))

def confusion_diff(before_cm, after_cm):
    return np.abs(before_cm - after_cm)
