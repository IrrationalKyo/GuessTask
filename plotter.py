import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join, abspath
import csv
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


''' this code is from tutorial'''
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=10,
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def confusion_parser(file_dir):
    with open(file_dir, 'r') as f:
        reader = csv.reader(f)
        return np.array(list(reader),dtype=np.float32)
        # print(list(reader))

def average_confusion(list_of_confusion):
    total_confusion = np.zero(list_of_confusion[0].shape)


# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()

if __name__ == "__main__":
    directory = "./result_varification"
    onlyfiles = [f for f in listdir(abspath(directory)) if isfile(join(abspath(directory), f))]
    for f in onlyfiles:
        fl = f.split(".")
        if fl[-1] == str("confusion"):
            cnf_mat = confusion_parser(directory+"/"+f)
            plt.figure(figsize=(10,10), dpi=100)
            plot_confusion_matrix(cnf_mat, classes=range(16), normalize=True,
                                  title='Normalized confusion matrix')

            plt.savefig(directory+"/"+fl[0]+"normalized.png")
            plt.figure(figsize=(10, 10), dpi=100)
            plot_confusion_matrix(cnf_mat.astype(int), classes=range(16), normalize=False,
                                  title='Non-Normalized confusion matrix')

            plt.savefig(directory+"/"+fl[0] + ".png")