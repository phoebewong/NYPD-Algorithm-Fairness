import matplotlib.pyplot as plt
import numpy as np
import utils
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

def plot_cm(y_true, y_pred, classes,name,ax,cmap=plt.cm.Blues):

    title = 'Confusion Matrix for {}'.format(name)
    cm = confusion_matrix(y_true, y_pred)
    ax.grid(False)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Predicted label', fontsize=15)
    ax.set_ylabel('True label', fontsize=15)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=20)
    return ax

def plot_feature_imp(X_tr, model, name, ax, k=10):
    indices, features, importances = utils.top_k_features(X_tr.columns, model.feature_importances_, k)

    ax.barh(range(len(indices)), importances, color='b', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Relative Importance', fontsize=15)
    ax.set_title('Top {} Features of {}'.format(k, name), fontsize=20)
    return ax
