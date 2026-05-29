import numpy as np
from matplotlib import pyplot as plt

def _groundTruthMatrix(y_true, labels):
    """
    Return the ground truth matrix from a given array of true values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    Returns
    ----------
    groundTruthM : ndarray of shape (n_samples, n_classes).
        Ground truth matrix.
    """
    # Size of the test
    size = y_true.shape[0]

    # Initializes some utility arrays
    groundTruthM = np.zeros((size, len(labels)))
    index = np.array([])

    # Makes a binary array from y_true
    for i in range(size):
        class_y = y_true[i]
        for j in range(len(labels)):
            if(labels[j] == class_y):
                index = np.append(index, np.where(labels == labels[j])[0])

    # Turns the binary array into a binary matrix
    groundTruthM[np.arange(size), index.astype(int)] = 1

    return groundTruthM

def prob_confusion_matrix(y_true, y_score, labels=None, abs_tolerance=1e-8):
    """
    Compute the probabilistic confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1.0, default = 1e-8.

    Returns
    ----------
    prob_conf_matrix : ndarray of shape (n_classes, n_classes).
        Probabilistic confusion matrix.
    """
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
    
    # Checks input values
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples.") 
    elif len(np.unique(y_true)) != y_score.shape[1]:
        raise ValueError("'y_true' and 'y_score' have different number of classes.")
    elif not np.allclose(1, y_score.sum(axis=1), rtol=0, atol=abs_tolerance):
        raise ValueError(
            "Target scores need to be probabilities and they should sum up to 1.0 over classes."
        )
    
    # Checks labels list
    if labels is None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif not set(y_true).issubset(set(labels)):
            raise ValueError("All of the labels in 'y_true' must be in 'labels'.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    # Transforms y_test into a proper ground truth matrix
    groundTruthM = _groundTruthMatrix(y_true, labels)

    # Computes the probabilistic confusion matrix
    prob_conf_matrix = np.dot(np.transpose(groundTruthM), y_score)

    return prob_conf_matrix

def confidence_correctness_matrix(y_true, y_score, labels=None, abs_tolerance=1e-8, class_dependent=False):
    """
    Calculate the confidence-correctness matrix in its class-independent or class-dependent versions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1.0. Default = 1e-8.

    class-dependent : boolean, default=False.
        
        If 'False', the class-independent confidence-correctness matrix is computed, and
        if 'True', the class-dependent confidence-correctness matrix is computed.

    Returns
    ----------
    confCorrM : pandas DataFrame.
        Returns the values of the serendipity matrix.
    """
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples.")
    
    if len(np.unique(y_true)) != y_score.shape[1]:
        raise ValueError("'y_true' and 'y_score' have different number of classes.")
    
    if not np.allclose(1, y_score.sum(axis=1), rtol=0, atol=abs_tolerance):
        raise ValueError(
            "Target scores need to be probabilities and they should sum up to 1.0 over classes."
        )
    
    # Checks i nput data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
    
    # Checks labels list
    if labels is None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif not set(y_true).issubset(set(labels)):
            raise ValueError("All of the labels in 'y_true' must be in 'labels'.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    confidence = ["Reliability","Overconfidence","Underconfidence","Ambiguity"]

    if class_dependent == False:
        tam = y_true.shape[0]

        # Decomposes the y_score probabilistic matrix into certainty and uncertainty
        H, L = confidence_matrices(y_true, y_score)

        # Computes the serendipity matrix
        H_acc = np.trace(H)
        H_inacc = np.sum (H) - np.trace(H)
        L_inacc = np.trace(L)
        L_acc = np.sum (L) - np.trace(L)

        serendipityM = np.array([H_acc, H_inacc, L_acc, L_inacc])
        serendipityM[serendipityM < 0] = 0

        serendipityM = dict(zip(confidence, serendipityM/tam))
    else:
        # Decomposes the y_score probabilistic matrix into certainty and uncertainty
        H, L = confidence_matrices(y_true, y_score)
          
        serendipityM = {key: {} for key in labels}

        # Computes the serendipity matrix by class
        for i in range(H.shape[0]):
            sum = np.sum(H[i,:], axis=0) + np.sum(L[i,:], axis=0)
            
            if sum != 0:
                serendipityM[labels[i]]["Reliability"] = H[i][i]/sum
                serendipityM[labels[i]]["Overconfidence"] = (np.sum(H[i,:], axis=0) - H[i][i])/sum
                serendipityM[labels[i]]["Ambiguity"] = L[i][i]/sum
                serendipityM[labels[i]]["Underconfidence"] = (np.sum(L[i,:], axis=0) - L[i][i])/sum

    return serendipityM


def confidence_matrices(y_true, y_score, labels=None, abs_tolerance=1e-8):
    """
    Compute the high-confidence and low-confidence matrices.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1.0, default = 1e-8.

    Returns
    ----------
    H : ndarray of shape (n_classes, n_classes).
        High-confidence matrix.

    L : ndarray of shape (n_classes, n_classes).
        Low-confidence matrix.
    """
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
    
    # Checks input values
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples.") 
    elif len(np.unique(y_true)) != y_score.shape[1]:
        raise ValueError("'y_true' and 'y_score' have different number of classes.")
    elif not np.allclose(1, y_score.sum(axis=1), rtol=0, atol=abs_tolerance):
        raise ValueError(
            "Target scores need to be probabilities and they should sum up to 1.0 over classes."
        )
    
    # Checks label list
    if labels is None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif not set(y_true).issubset(set(labels)):
            raise ValueError("All of the labels in 'y_true' must be in 'labels'.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    # Transforms y_test into a proper ground truth matrix
    groundTruthM = _groundTruthMatrix(y_true, labels)

    # Initializes two empty matrix and one empty array
    H_confM = np.zeros((y_score.shape[0], y_score.shape[1]))
    L_confM = np.zeros((y_score.shape[0], y_score.shape[1]))
    classes = np.zeros((y_score.shape[1]))

    # Fills the empty array with a different number for each different class
    for i in range(1, y_score.shape[1]):
        classes[i] = i

    # Saves the index for the most likely prediction from the probabilistic prediction matrix
    H_confIndex = np.argmax(y_score, axis=1)

    # Saves the probabilistic predictions into the high-confidence and low-confidence matrices
    for i in range(y_score.shape[0]):
        L_confIndex = np.delete(classes.astype(int), H_confIndex[i])
        H_confM[i][H_confIndex[i]] = np.max(y_score[i])
        for j in range(len(L_confIndex)):
            L_confM[i][L_confIndex[j]] = y_score[i][L_confIndex[j]]

    # Computes the high-confidence and low-confidence matrices
    H = np.dot(np.transpose(groundTruthM), H_confM)
    L = np.dot(np.transpose(groundTruthM), L_confM)

    return H, L

def confidence_weights(y_true, y_score, labels=None):
    """
    Calculate the lambda values for the high-confidence and low-confidence matrices.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    Returns
    ----------
    lambda_H : float.
        Lambda value for the high-confidence matrix.

    lambda_L : float.
        Lambda value for the low-confidence matrix.
    """
    # Computes the probabilistic confusion matrix 
    prob_conf_matrix = prob_confusion_matrix(y_true, y_score, labels)

    # Computes the high-confidence and low-confidence matrices 
    H, L = confidence_matrices(y_true, y_score, labels)

    # Calculates lambda values for the high-confidence and low-confidence matrices
    lambda_H = np.sum(H)/np.sum(prob_conf_matrix)
    lambda_L = np.sum(L)/np.sum(prob_conf_matrix)

    return lambda_H, lambda_L

def prob_accuracy_score(y_true, y_score, labels=None):
    """
    Compute the probabilistic accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    Returns
    ----------
    prob_acc : float.
        Probabilistic accuracy.
    """
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    # Computes the probabilistic confusion matrix
    prob_conf_matrix = prob_confusion_matrix(y_true, y_score, labels)
    
    # Calculates the probabilistic accuracy
    TP_sum = np.sum(np.diag(prob_conf_matrix))

    prob_acc = np.divide(TP_sum, np.sum(prob_conf_matrix), where=np.sum(prob_conf_matrix) != 0, out=np.zeros((prob_conf_matrix.shape[0])))[0]

    return prob_acc

def prob_balanced_accuracy_score(y_true, y_score, labels=None):
    """
    Compute the probabilistic balanced accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    Returns
    ----------
    prob_b_acc : float.
        Probabilistic balanced accuracy.
    """
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    # Computes the probabilistic confusion matrix
    prob_conf_matrix = prob_confusion_matrix(y_true, y_score, labels)

    # Checks labels list
    if labels is None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif not set(y_true).issubset(set(labels)):
            raise ValueError("All of the labels in 'y_true' must be in 'labels'.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    # Calculates the probabilistic balanced accuracy
    TP = np.diag(prob_conf_matrix) 
    FN = np.sum(prob_conf_matrix, axis=1) - TP

    sensitivity = np.divide(TP, TP + FN, where=TP + FN != 0, out=np.zeros((prob_conf_matrix.shape[0])))

    prob_b_acc = np.mean(sensitivity)

    return prob_b_acc

def prob_cohen_kappa_score(y_true, y_score, labels=None):
    """
    Compute the probabilistic Cohen Kappa.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    Returns
    ----------
    prob_cohen_kappa : float.
        Probabilistic Cohen Kappa.
    """
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    # Checks labels list
    if labels == None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    # Calculates the Cohen Kappa
    y_pred = np.argmax(y_score, axis=1)
    y_pred = np.take(np.unique(y_true), y_pred)

    p_o = 0
    p_e = 0
    p_aux = np.zeros((len(labels),2))

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            p_o += (1 + y_score[i][list(labels).index(y_true[i])])/2

        p_aux[list(labels).index(y_true[i])][0] += 1
        p_aux[list(labels).index(y_true[i])][1] += y_score[i][list(labels).index(y_true[i])]
    
    p_o /= len(y_true)
    p_aux /= len(y_true)

    for i in range(len(labels)):
        p_e += p_aux[i][0] * p_aux[i][1]

    prob_cohen_kappa = (p_o - p_e)/(1 - p_e)

    return prob_cohen_kappa

def prob_matthews_corrcoef(y_true, y_score, labels=None):
    """
    Compute the probabilistic Matthews Correlation Coefficient.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    Returns
    ----------
    prob_m_corrcoef : float.
        Probabilistic Matthews Correlation Coefficient.
    """
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    # Computes the probabilistic confusion matrix
    prob_conf_matrix = prob_confusion_matrix(y_true, y_score, labels)

    # Calculates the MCC
    TP = np.diag(prob_conf_matrix) 
    FN = np.sum(prob_conf_matrix, axis=0) - TP
    FP = np.sum(prob_conf_matrix, axis=1) - TP
    TN = np.sum(prob_conf_matrix) - (TP + FN + FP) 

    prob_m_corrcoef = np.divide(TP*TN-FP*FN, np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)), where=(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) != 0, out=np.zeros((prob_conf_matrix.shape[0])))[0]

    return prob_m_corrcoef

def prob_precision_score(y_true, y_score, labels=None, pos_label=1, average="binary"):
    """
    Compute the probabilistic precision.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    pos_label : int, default=1.
        The class to report if average='binary' and the data is binary, 
        otherwise this parameter is ignored.

    average : string, default="binary".
        This parameter is required for multiclass targets and determines the type of averaging 
        performed on the data: "binary", "micro", "macro" and "weighted".

    ----------
    prob_prec : float.
        Probabilistic precision.
    """
    # Checks average
    if average not in ["binary", "micro", "macro", "weighted"]:
            raise ValueError("'average' should be 'binary' for binary targets or 'macro', 'micro' or 'weigthed' for multiclass targets.")
   
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    # Checks labels list
    if labels == None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    if len(labels) > 2 and average == "binary":
        raise ValueError("For multiclass targets 'average' should be 'macro', 'micro' or 'weigthed'.")
    elif len(labels) == 2 and not average == "binary":
        raise ValueError("For binary targets 'average' should be 'binary'.")

    # Computes the probabilistic confusion matrix
    prob_conf_matrix = prob_confusion_matrix(y_true, y_score, labels)

    # Calculates the probabilistic precision
    TP = np.diag(prob_conf_matrix) 
    FP = np.sum(prob_conf_matrix, axis=0) - TP

    if average == "binary":
        prob_prec = np.divide(TP, TP + FP, where=TP + FP != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        return prob_prec[pos_label]
    elif average == "micro":
        if not np.sum(TP) + np.sum(FP) == 0:
            prob_prec = np.sum(TP)/(np.sum(TP) + np.sum(FP))

            return prob_prec
        else:
            return 0
    elif average == "macro":
        prob_prec = np.divide(TP, TP + FP, where=TP + FP != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        return np.mean(prob_prec)
    elif average == "weighted":
        prob_prec = np.divide(TP, TP + FP, where=TP + FP != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        weights = []
        for i in range(len(labels)):
            weights.append(list(y_true).count(labels[i])/len(y_true))

        prob_prec = prob_prec*np.asarray(weights)

        return np.sum(prob_prec)

def prob_recall_score(y_true, y_score, labels=None, pos_label=1, average="binary"):
    """
    Compute the probabilistic recall.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    pos_label : int, default=1.
        The class to report if average='binary' and the data is binary, 
        otherwise this parameter is ignored.

    average : string, default="binary".
        This parameter is required for multiclass targets and determines the type of averaging 
        performed on the data: "binary", "micro", "macro" and "weighted".

    Returns
    ----------
    prob_rec : float.
        Probabilistic recall.
    """
    # Checks average
    if average not in ["binary", "micro", "macro", "weighted"]:
            raise ValueError("'average' should be 'binary' for binary targets or 'macro', 'micro' or 'weigthed' for multiclass targets.")
    
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    # Checks labels list
    if labels == None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    if len(labels) > 2 and average == "binary":
        raise ValueError("For multiclass targets 'average' should be 'macro', 'micro' or 'weigthed'.")
    elif len(labels) == 2 and not average == "binary":
        raise ValueError("For binary targets 'average' should be 'binary'.")

    # Computes the probabilistic confusion matrix
    prob_conf_matrix = prob_confusion_matrix(y_true, y_score, labels)

    # Calculates the probabilistic recall
    TP = np.diag(prob_conf_matrix) 
    FN = np.sum(prob_conf_matrix, axis=1) - TP

    
    if average == "binary":
        prob_rec = np.divide(TP, TP + FN, where=TP + FN != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        return prob_rec[pos_label]
    elif average == "micro":
        if not np.sum(TP) + np.sum(FN) == 0:
            prob_rec = np.sum(TP)/(np.sum(TP) + np.sum(FN))

            return prob_rec
        else:
            return 0
    elif average == "macro":
        prob_rec = np.divide(TP, TP + FN, where=TP + FN != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        return np.mean(prob_rec)
    elif average == "weighted":
        prob_rec = np.divide(TP, TP + FN, where=TP + FN != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        weights = []
        for i in range(len(labels)):
            weights.append(list(y_true).count(labels[i])/len(y_true))

        prob_rec = prob_rec*np.asarray(weights)

        return np.sum(prob_rec)

def prob_f1_score(y_true, y_score, labels=None, pos_label=1, average="binary"):
    """
    Compute the probabilistic F1-score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). 
        Ground truth (correct) labels.

    y_score : array-like of shape (n_samples, n_classes).
        Probabilities of predicted labels, as returned by a classifier. The sum of 
        these probabilities must sum up to 1.0 over classes.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

    labels : array-like of shape (n_classes,), default=None.
        List of labels to index the matrix. 
        
        If 'None' is given, those that appear at least once
        in 'y_true' or 'y_pred' are used in sorted order.

    pos_label : int, default=1.
        The class to report if average='binary' and the data is binary, 
        otherwise this parameter is ignored.

    average : string, default="binary".
        This parameter is required for multiclass targets and determines the type of averaging 
        performed on the data: "binary", "micro", "macro" and "weighted".

    Returns
    ----------
    prob_f1 : float.
        Probabilistic F1-score.
    """
    # Checks average
    if average not in ["binary", "micro", "macro", "weighted"]:
            raise ValueError("'average' should be 'binary' for binary targets or 'macro', 'micro' or 'weigthed' for multiclass targets.")
    
    # Checks input data type
    if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

    # Checks labels list
    if labels == None:
        labels = np.unique(y_true)
    else:
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in 'y_true'.")
        
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

    if len(labels) > 2 and average == "binary":
        raise ValueError("For multiclass targets 'average' should be 'macro', 'micro' or 'weigthed'.")
    elif len(labels) == 2 and not average == "binary":
        raise ValueError("For binary targets 'average' should be 'binary'.")

    # Computes the probabilistic confusion matrix
    prob_conf_matrix = prob_confusion_matrix(y_true, y_score, labels)

    # Calculates the probabilistic balanced accuracy
    TP = np.diag(prob_conf_matrix) 
    FN = np.sum(prob_conf_matrix, axis=1) - TP
    FP = np.sum(prob_conf_matrix, axis=0) - TP

    if average == "binary":
        prob_f1 = np.divide(2*TP, 2*TP + FP + FN, where=TP + FP + FN != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        return prob_f1[pos_label]
    elif average == "micro":
        if not np.sum(2*TP) + np.sum(FP) + np.sum(FN) == 0:
            prob_f1 = np.sum(2*TP)/(np.sum(2*TP) + np.sum(FP) + np.sum(FN))

            return prob_f1
        else:
            return 0
    elif average == "macro":
        prob_f1 = np.divide(2*TP, 2*TP + FP + FN, where=TP + FP + FN != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        return np.mean(prob_f1)
    elif average == "weighted":
        prob_f1 = np.divide(2*TP, 2*TP + FP + FN, where=TP + FP + FN != 0, out=np.zeros((prob_conf_matrix.shape[0])))

        weights = []
        for i in range(len(labels)):
            weights.append(list(y_true).count(labels[i])/len(y_true))

        prob_f1 = prob_f1*np.asarray(weights)

        return np.sum(prob_f1)
