import keras
import pydot_ng as pydot
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, TimeDistributed, Bidirectional, LSTM, RepeatVector
import numpy as np
from sklearn.metrics import confusion_matrix
import re
import converter as cvt
import math
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import json
import glob
import os
from pathlib2 import Path
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


'''
    weighted_categorical_crossentropy implementation by Mike Clark https://gist.github.com/wassname
'''
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

# TODO: CONVERTDICT TO NUMPY ARRAY
def get_priors(trace):
    prior_count = {}
    normalized = {}
    inverse = {}
    for i in trace:
        if i in prior_count:
            prior_count[i] += 1
        else:
            prior_count[i] = 1
    total = 0



    for key, value in prior_count.items():
        normalized[key] = float(value/len(trace))
        inverse[key] = float(len(trace)/value)

    inverse_output = np.zeros(len(inverse))
    for key, value in inverse.items():
        inverse_output[key] = math.log(value)

    return None, inverse_output

# turns trace into binary vector
def mask_trace(value, trace):
    for i in range(len(trace)):
        val = trace[i]
        if val == value:
            trace[i] = 1
        else:
            trace[i] = 0
    return trace

def uniform_sample_trace(sample_gap, trace):
    output_trace = []
    for i in range(len(trace)):
        if i%sample_gap == 1:
            output_trace.append(trace[i])
    return output_trace

def dict_mapper(dic):
    def sample_weight_mapper(vector):
        label = np.argmax(vector)
        return dic[label]
    return sample_weight_mapper


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


def create_model(cell_count, shape, stateful, batch, output_dim, loss="mse", drop_out = True, layers=1):
    model = Sequential()

    model.add(LSTM(cell_count,
          stateful=stateful,
          return_sequences=False, name="lstm_1", batch_size=batch, input_shape=shape)
          )
    model.add(RepeatVector(shape[0]))
    model.add(LSTM(math.ceil(cell_count/4),
                   stateful=stateful,
                   return_sequences=True, name="lstm_2", batch_size=batch)
              )
    model.add(TimeDistributed(Dense(output_dim, activation='linear', name="output_layer")))
    model.compile(loss=loss, optimizer="rmsprop", metrics=['accuracy'])
    return model

def manual_verification(model, test_dataset, label_card, batch_size=1):
    model.reset_states()
    y = model.predict(test_dataset[0], batch_size=batch_size)

    time_step = 1
    print("time_step" + str(time_step))
    true_pred_y = []
    true_y = []

    total_count = 0
    total_len = len(y)

    print(y)
    time_step = len(y[0])

    for k in range(time_step):
        for i in range(len(y)):
            pred_y = np.argmax(y[i][k])
            true_pred_y.append(pred_y)
            label = np.argmax(test_dataset[1][i][k])
            true_y.append(label)
    correct = 0

    print(pred_y)
    confusion = confusion_matrix(true_y, true_pred_y, labels=range(label_card))
    for i in range(label_card):
        correct += confusion[i][i]
    acc = float(correct / len(y))

    print(total_count/total_len)

    return (confusion, acc)

def self_predict(model, test_dataset, prediction_length, label_card, batch_size=1):
    model.reset_states()

    predictions = []
    start = test_dataset[0][:batch_size]
    for i in range(len(test_dataset[0])):
        y = model.predict(start, batch_size=batch_size)
        predictions.append(y)
        start = y
    print(predictions)
    exit()

    return


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


def run_instance(data_name, cell_size = 32, layers = 1, epoch = 50, batch_size = 32, prediction_len = 1, offset = 0, validation = True, timesteps = 10, target_task = 0):
    # TODO:
    label_name = tasksize_extractor(data_name)
    rep_number = rep_extractor(data_name)
    label_card = 2

    result_path = "./result/" + str(label_name) + "_" + str(rep_number) + "/"
    modelname = result_path + str(cell_size) + "_" + str(layers) + "_" + str(epoch) + "_" + str(batch_size) + "_" + str(
        prediction_len) + "_" + str(offset) + ".model"
    statname = result_path + str(cell_size) + "_" + str(layers) + "_" + str(epoch) + "_" + str(batch_size) + "_" + str(
        prediction_len) + "_" + str(offset) + ".json"

    if file_exists(modelname):
        return None

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    trace = cvt.newText_to_list(data_name)
    trace= mask_trace(target_task,trace)
    trace = uniform_sample_trace(100, trace)

    prior, inverse = get_priors(trace)


    for t in trace:
        if not isinstance(t, int):
            print("Not parsing correctly {}".format(t))


    example = cvt.list_to_example_sequence_binary(trace, label_card, offset=offset + timesteps, pred_len=prediction_len, seed=data_name)


    dataset1 = cvt.chunk_examples(example[0], example[1], 0, len(example[0]))
    print(np.asarray(dataset1[0]))

    train_x, train_y, test_x, test_y = cvt.split_train_test(0.80, example)
    total_len = len(train_x)
    total_len -= total_len % (batch_size * timesteps)
    train_x = train_x[:total_len]
    train_y = train_y[:total_len]

    # sample_weights = map(dict_mapper(prior), train_x)
    # sample_weights = np.reshape(list(sample_weights), (math.floor(total_len/timesteps), timesteps))


    train_x = np.reshape(train_x, (math.floor(total_len/timesteps), timesteps, 1))
    train_y = np.reshape(train_y, (math.floor(total_len/timesteps), timesteps, 1))

    print(len(train_x))
    print(len(train_y))

    validation_ratio = None

    if validation:
        validation_len = (total_len * 0.1)
        validation_ratio = (validation_len - (validation_len % (batch_size * timesteps))) / total_len

    print(validation_ratio)

    total_len = len(test_x)
    total_len -= total_len % (batch_size * timesteps)
    test_x = test_x[:total_len]
    test_y = test_y[:total_len]
    test_x = np.reshape(test_x, (math.floor(total_len/timesteps), timesteps, 1))
    test_y = np.reshape(test_y, (math.floor(total_len/timesteps), timesteps, 1))

    wcc = weighted_categorical_crossentropy(inverse)

    model = create_model(cell_size, (train_x.shape[1], train_x.shape[2]), stateful=True,
                         batch=batch_size,
                         output_dim=prediction_len*label_card,
                         layers=layers,
                         loss="mse")

    for i in range(1):
        model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, verbose=2, validation_split=validation_ratio)
        model.reset_states()
        print("Current epoch: " + str(i))

    # TODO:
    model.save(modelname)

    scores = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=2)
    model.reset_states()
    confusion, accuracies = manual_verification(model, (test_x, test_y), label_card, batch_size=batch_size)
    normal = confusion

    statJSON = {}
    statJSON["accuracy"] = scores[1]
    statJSON["normalized_matrix"] = {}
    for i in range(label_card):
        statJSON["normalized_matrix"][i] = normal[i].tolist()
    statJSON["matrix"] = {}
    for i in range(label_card):
        statJSON["matrix"][i] = confusion.astype("float")[i].tolist()

    save_result(statname, statJSON)

    return scores, confusion, normal, model


if __name__ == "__main__":



    layer_set = set([3])
    cell_set = set([128])
    offset_set = set([0])

    fileNames = glob.glob('./data/*.data')

    for filename in fileNames:
        for layer in layer_set:
            for cell in cell_set:
                for offset in offset_set:

                    info = run_instance(filename, cell_size=cell, layers=layer, epoch=100, batch_size=10, prediction_len=1, offset=10, timesteps=10, target_task=4)

                    if info is None:
                        continue
                    normal_mat = info[2]
                    scores = info[0]
                    print(normal_mat)
                    print(scores)