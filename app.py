import keras
import taskrecon_converter as cvt
import taskrecon_guesser as gue
import os
import plotter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import math
from pathlib2 import Path
import numpy as np
import json
import argparse
import gc



def file_exists(fileName):
    theFile = Path(fileName)
    if theFile.is_file():
        return True
    return False


def save_result(fileName, data_dict):
    with open(fileName, "w") as file:
        file.write(json.dumps(data_dict, indent=4, sort_keys=True))
    return


def generate_single_fold(fold_count, data_pair, batch_size=1):
    example_list = data_pair[0]
    label_list = data_pair[1]
    partition = fold_count + 1
    if len(example_list) != len(label_list):
        raise ValueError("expected example and label to have same number of samples")

    partition_size = math.floor(len(example_list) / partition)
    partition_size -= partition_size % batch_size

    ''' list of lists. each element of result is a list of partition where the last parition is the testing set'''
    result = []

    fold = []
    fold.append(cvt.chunk_examples(example_list, label_list, 0, (fold_count) * partition_size))
    fold.append(
        cvt.chunk_examples(example_list, label_list, (fold_count) * partition_size, (fold_count + 1) * partition_size))
    result.append(fold)

    return result


if __name__ == "__main__":

    gc.enable()

    """
    parser = argparse.ArgumentParser(description="Parameters for running keras")
    parser.add_argument("-n", "--nsize", help="number of lstm cells perlayer (Default 50)")
    parser.add_argument("-b", "--batchsize", help="size of batch (Default 200)")
    parser.add_argument("-e", "--epoch", help="number of epoch for training (Default 10)")
    parser.add_argument("-t", "--timesteps", help="number of timesteps considered for training/predicting (Default 100)")
    parser.add_argument("-g", "--gap", help="the amount of time between input vector X and its label Y (Default 0)")
    parser.add_argument("-d", "--dropout", help="0 to turn off dropout layer. 1 to turn on (Default 1)")
    parser.add_argument("-f", "--fold", help="1/(value+1) of the data will be used for test, rest for training (Default 4)")
    """

    result_list = []

    fileNames = glob.glob('./data/*.data')
    loss = ["categorical_crossentropy"]
    node_size = [100]
    batch_size = [1000]
    epoch = [7]
    time_steps = [100]
    gap = [False]
    drop_out = [True]
    fold_count = 4

    iteration = 0

    for name in fileNames:
        data_list = cvt.newText_to_list(name)[0:200000]
        for t in time_steps:
            for l in loss:
                for d in drop_out:
                    for b in batch_size:
                        for e in epoch:
                            for ga in gap:
                                g = 1
                                if ga:
                                    g = 50
                                for n in node_size:
                                    n_size = n
                                    b_size = b
                                    time = t
                                    ep = e
                                    out = d
                                    if l == "poisson":
                                        lo = "poi"
                                    else:
                                        lo = "ca"
                                    id = name.split("/")[-1].split(".")[0] + ".result"
                                    directory = "./result/" + id
                                    fileName = lo + "_lstm_lstm_fold_n" + str(n_size) + "_e" + str(ep) + "_g" + str(g)
                                    modelName = directory + "/" + fileName + ".model"
                                    statName = directory + "/stat.json"

                                    print("Training: " + str(name) + "\tid: " + str(id))

                                    if file_exists(modelName) and file_exists(statName):
                                        continue

                                    if iteration >= 10:
                                        exit()

                                    print("Going to GENERATE")
                                    folds = generate_single_fold(fold_count,
                                                                 cvt.list_to_example_overlap(data_list, time_steps=time,
                                                                                             overlap_gap=g),
                                                                 batch_size=b_size)

                                    i = 1
                                    iteration += 1

                                    fold = folds[0]
                                    x_train = fold[0][0]
                                    y_train = fold[0][1]
                                    x_test = fold[1][0]
                                    y_test = fold[1][1]

                                    class_card = len(x_train[0][0])

                                    if not file_exists(modelName):
                                        print("Going to CREATE MODEL")
                                        model = gue.create_model(n_size, (time, class_card), stateful=True,
                                                                 batch=b_size,
                                                                 output_dim=class_card, loss=l, drop_out=out)
                                        model.fit(x_train, y_train, epochs=ep, batch_size=b_size, verbose=1)
                                        if not os.path.exists(directory):
                                            os.makedirs(directory)
                                        model.save(modelName)
                                    else:
                                        model = keras.models.load_model(modelName)

                                    i += 1

                                    if g > 1:
                                        (cnf_mat, acc) = gue.manual_verification_disjoint(model, (x_test, y_test),
                                                                                          batch_size=b_size)
                                    else:
                                        # print(model.evaluate(x_test, y_test, batch_size=batch_size))
                                        (cnf_mat, acc) = gue.manual_verification_100(model, (x_test, y_test),
                                                                                     batch_size=b_size)

                                    '''
                                    plt.figure(figsize=(10, 10), dpi=100)
                                    plotter.plot_confusion_matrix(cnf_mat, classes=range(class_card),
                                                                  normalize=True,
                                                                  title='Normalized confusion matrix')

                                    plt.savefig(directory + "/" + fileName + "_normalized_" + str(
                                        acc) + ".png")
                                    plt.figure(figsize=(10, 10), dpi=100)
                                    plotter.plot_confusion_matrix(cnf_mat.astype(int), classes=range(class_card),
                                                                  normalize=False,
                                                                  title='Non-Normalized confusion matrix')

                                    plt.savefig(
                                        directory + "/" + fileName + "_" + str(acc) + ".png")
                                    '''

                                    normal_cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

                                    statJSON = {}
                                    statJSON["accuracy"] = acc
                                    statJSON["normalized_matrix"] = {}
                                    for i in range(class_card):
                                        statJSON["normalized_matrix"][i]=normal_cnf_mat[i].tolist()
                                    statJSON["matrix"] = {}
                                    for i in range(class_card):
                                        statJSON["matrix"][i]= cnf_mat.astype("float")[i].tolist()

                                    '''
                                    for i in range(class_card):
                                        statString += "task_" + str(i) + ":" + str(normal_cnf_mat[i][i]) + "\n"


                                    save_result(directory + "/" + "stat.json",
                                                "accuracy:" + str(acc) + "\n" + statString)
                                    '''
                                    save_result(directory + "/" + "stat.json", statJSON)


                                    fold = None
                                    x_train = None
                                    y_train = None
                                    x_test = None
                                    y_test = None

                                    gc.collect()

