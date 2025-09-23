from sklearn.metrics import roc_auc_score

def calculate_auroc(confidences, correctness):
    """
    Computes the Area Under the Receiver Operating Characteristic curve (AUROC).

    Args:
        confidences (list or numpy.ndarray): Model confidence scores for each prediction.
            Higher values should indicate higher likelihood of correctness.
        correctness (list or numpy.ndarray): Boolean or {0,1} labels indicating whether
            each prediction was correct.

    Returns:
        float: AUROC score in [0, 1]. 0.5 indicates random chance, 1.0 is perfect.
    """
    return roc_auc_score(correctness, confidences)