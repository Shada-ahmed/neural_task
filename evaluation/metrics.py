
import numpy as np


def confusion_matrix_manual(y_true,y_pred):
    """
    Build a binary confusion matrix for labels -1 and +1.

    Rows represent actual classes.
    Columns represent predicted classes.

    Order: -1, +1
    """
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)

    if len(y_true)!=len(y_pred):
        raise ValueError("Actual and predicted labels must have equal lengths.")

    matrix=np.zeros((2,2),dtype=int)

    for actual,predicted in zip(y_true,y_pred):
        if actual==-1 and predicted==-1:
            matrix[0,0]+=1
        elif actual==-1 and predicted==1:
            matrix[0,1]+=1
        elif actual==1 and predicted==-1:
            matrix[1,0]+=1
        elif actual==1 and predicted==1:
            matrix[1,1]+=1
        else:
            raise ValueError("Labels must be -1 or +1.")

    return matrix


def accuracy_score_manual(y_true,y_pred):
    """Calculate classification accuracy manually."""
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)

    if len(y_true)==0:
        raise ValueError("Cannot calculate accuracy on empty data.")

    if len(y_true)!=len(y_pred):
        raise ValueError("Actual and predicted labels must have equal lengths.")

    correct=0

    for actual,predicted in zip(y_true,y_pred):
        if actual==predicted:
            correct+=1

    return correct/len(y_true)*100
