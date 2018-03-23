import keras
import pydot_ng as pydot
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNGRU, Embedding, Flatten, GRU, LSTM, CuDNNLSTM
import numpy as np
from sklearn.metrics import confusion_matrix
import re
import converter as cvt
from keras.utils import plot_model
import json
import math
import glob
import os
from pathlib2 import Path

def save_result(fileName, data_dict):
    with open(fileName, "w") as file:
        file.write(json.dumps(data_dict, indent=4, sort_keys=True))
    return

def score_display(fold_scores):
    fold_count = len(fold_scores)
    avg = []
    for i in range(fold_count):
        print("fold " + str(i) + ":", end="")
        fold_avg = []
        for j in range(fold_count-1):
            fold_avg.append(fold_scores[i][j][1])
            print("\t"+str(fold_scores[i][j][1]), end="")
        avg.append(np.mean(fold_avg))
        print("\tfold_avg: "+str(avg[i]))
    total_avg = np.mean(avg)
    print("total_avg: " + str(total_avg))
    return total_avg


def create_model(cell_count, shape, batch, label_card, output_dim, stateful= True, loss="categorical_crossentropy", drop_out = True, layers=1, timesteps = 1):
    model = Sequential()
    model.add(Embedding(label_card, 3, batch_size=batch, input_length=timesteps))
    model.add(CuDNNLSTM(cell_count,
              stateful=stateful, batch_size= batch,
			  return_sequences=True, name="lstm_1",
              kernel_regularizer=keras.regularizers.l2(0.01),
              activity_regularizer=keras.regularizers.l1(0.01)))
    for i in range(1, layers-1):
        model.add(CuDNNLSTM(cell_count,
                            input_shape=shape,
                            batch_size=batch,
                            stateful=stateful,
                            return_sequences=True, name="lstm_" + str(i+1),
                            kernel_regularizer=keras.regularizers.l2(0.01),
                            activity_regularizer=keras.regularizers.l1(0.01)))
    model.add(CuDNNLSTM(cell_count,
                       input_shape=shape,
                       batch_size=batch,
                       stateful=stateful,
                       return_sequences=False, name="lstm_last"))
    model.add(Dense(cell_count, activation='tanh'))
    if drop_out:
        model.add(Dropout(0.5))

    model.add(Dense(output_dim, activation='softmax'))
    rms = keras.optimizers.RMSprop(lr=0.002)

    model.compile(loss=loss, optimizer="adam", metrics=['accuracy'])
    return model

def manual_verification(model, test_dataset, label_card, batch_size=1):
    model.reset_states()
    y = model.predict(test_dataset[0], batch_size=batch_size)
    # otuput shape will be the same as the input shape
    time_step = len(y[1])
    print("time_step" + str(time_step))
    true_pred_y = []
    true_y = []
    total_len = len(y) - math.floor(len(y)/2)
    for i in range(math.floor(len(y)/2) , len(y)):
        pred_y = np.argmax(y[i])
        true_pred_y.append(pred_y)
        label = np.argmax(test_dataset[1][i])
        true_y.append(label)

    confusion = confusion_matrix(true_y, true_pred_y , labels=range(label_card))

    correct = 0
    for i in range(label_card):
        correct+=confusion[i][i]
    print("correct count: " + str(correct))
    print("acc: " + str(correct / total_len))

    return (confusion, float(correct / total_len))

def save_matrix(matrix, filename):
    with open(filename, "w") as file:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                file.write(str(matrix[i][j]))
                if j != len(matrix[0])-1:
                    file.write(",")
            if i != len(matrix)-1:
                file.write("\n")
    return

def tasksize_extractor(name):
    splits = re.split('(\d+)', name)
    return int(splits[1])
def rep_extractor(name):
    splits = re.split('(\d+)', name)
    return int(splits[3])
def file_exists(fileName):
    theFile = Path(fileName)
    if theFile.is_file():
        return True
    return False



def run_instance(data_name, cell_size = 32, layers = 1, epoch = 50, batch_size = 32, prediction_len = 1, offset = 0, validation = True, timesteps=1):
    # TODO:
    label_card = tasksize_extractor(data_name) + 1
    rep_number = rep_extractor(data_name)
    label_name = label_card - 1

    result_path = "./result/" + str(label_name) +"_" +str(rep_number) + "/"
    modelname = result_path + str(cell_size) + "_" + str(layers) + "_" + str(epoch) + "_" + str(batch_size) + "_" + str(prediction_len) + "_" + str(offset) + ".model"
    statname = result_path + str(cell_size) + "_" + str(layers) + "_" + str(epoch) + "_" + str(batch_size) + "_" + str(prediction_len) + "_" + str(offset) + ".json"

    if file_exists(modelname):
        return None

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    trace = cvt.newText_to_list(data_name)

    for t in trace:
        if not isinstance(t, int):
            print("Not parsing correctly {}".format(t))

    label_card = tasksize_extractor(data_name) + 1

    example = cvt.list_to_example_sequence(trace, label_card, offset=offset, pred_len=prediction_len, seed=data_name, timesteps=timesteps)

    for t in example[1]:
        if len(t) != label_card * prediction_len:
            print("THIS OFFENDS ME {}".format(t))

    dataset1 = cvt.chunk_examples(example[0], example[1], 0, len(example[0]))
    print(np.asarray(dataset1[0]))

    train_x, train_y, test_x, test_y = cvt.split_train_test(0.80, example)
    total_len = len(train_x)
    total_len -= total_len % batch_size
    train_x = train_x[:total_len - total_len %timesteps]
    train_y = train_y[:total_len]
    train_x = np.reshape(train_x, (total_len, timesteps))
    train_y = np.reshape(train_y, (total_len, len(train_y[0])))

    validation_ratio = None

    if validation:
        validation_len = (total_len * 0.1)
        validation_ratio = (validation_len - (validation_len % batch_size)) / total_len

    total_len = len(test_x)
    total_len -= total_len % batch_size
    test_x = test_x[:total_len - total_len %timesteps]
    test_y = test_y[:total_len]
    test_x = np.reshape(test_x, (total_len, timesteps))
    test_y = np.reshape(test_y, (total_len, len(test_y[0])))


    model = create_model(cell_size, train_x.shape, batch_size, label_card, stateful=True,
                         output_dim=label_card * prediction_len,
                         layers=layers,
                         loss="categorical_crossentropy", timesteps=timesteps)
    for i in range(epoch):
        model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, validation_split=validation_ratio)
        model.reset_states()
        print("Current epoch: " + str(i))

    # TODO:
    model.save(modelname)

    scores = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=2)
    model.reset_states()
    confusion, accuracies = manual_verification(model, (test_x, test_y), label_card, batch_size=batch_size)
    normal = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

    statJSON = {}
    statJSON["accuracy"] = scores[1]
    statJSON["normalized_matrix"] = {}
    for i in range(label_card):
        statJSON["normalized_matrix"][i] = normal[i].tolist()
    statJSON["matrix"] = {}
    for i in range(label_card):
        statJSON["matrix"][i] = confusion.astype("float")[i].tolist()

    save_result(statname, statJSON)

    return scores, confusion, normal


if __name__ == "__main__":

    layer_set = set([1,2,3])
    cell_set = set([32, 64, 100])
    offset_set = set([0, 5, 50, 100])

    fileNames = glob.glob('./data/*.data')

    for filename in fileNames:
        for layer in layer_set:
            for cell in cell_set:
                for offset in offset_set:

                    info = run_instance(filename, cell_size=cell, layers=layer, epoch=50, batch_size=100, prediction_len=1, offset=offset, timesteps=200)

                    if info is None:
                        continue
                    normal_mat = info[2]
                    scores = info[0]
                    print(normal_mat)
                    print(scores)

