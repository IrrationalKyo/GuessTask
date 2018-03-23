import keras
import pydot_ng as pydot
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNGRU, TimeDistributed
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
        inverse_output[key] = value

    return None, inverse_output


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


def create_model(cell_count, shape, stateful, batch, output_dim, loss="categorical_crossentropy", drop_out = True, layers=1):
    model = Sequential()
    model.add(CuDNNGRU(cell_count,
              input_shape=shape,
			  batch_size=batch,
              stateful=stateful,
			  return_sequences=True, name="lstm_1",
              ))
    for i in range(1, layers):
        model.add(CuDNNGRU(cell_count,
                            input_shape=shape,
                            batch_size=batch,
                            stateful=stateful,
                            return_sequences=True, name="lstm_" + str(i+1),
                            ))
    model.add(TimeDistributed(Dense(math.floor(cell_count / 2), activation='relu')))
    model.add(TimeDistributed(Dense(math.floor(cell_count / 8), activation='tanh')))
    if drop_out:
        model.add(Dropout(0.5))

    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    rms = keras.optimizers.RMSprop(lr=0.002, clipvalue=100)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'], sample_weight_mode="temporal")
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

    time_step = len(y[0])

    for k in range(time_step):
        for i in range(len(y)):
            pred_y = np.argmax(y[i][k])
            true_pred_y.append(pred_y)
            label = np.argmax(test_dataset[1][i][k])
            true_y.append(label)
    correct = 0


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


def run_instance(data_name, cell_size = 32, layers = 1, epoch = 50, batch_size = 32, prediction_len = 1, offset = 0, validation = True, timesteps = 10):
    # TODO:
    label_card = tasksize_extractor(data_name) + 1
    rep_number = rep_extractor(data_name)
    label_name = label_card - 1

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

    prior, inverse = get_priors(trace)


    for t in trace:
        if not isinstance(t, int):
            print("Not parsing correctly {}".format(t))


    example = cvt.list_to_example_sequence(trace, label_card, offset=offset + timesteps, pred_len=prediction_len, seed=data_name)

    for t in example[1]:
        if len(t) != label_card * prediction_len:
            print("THIS OFFENDS ME {}".format(t))

    dataset1 = cvt.chunk_examples(example[0], example[1], 0, len(example[0]))
    print(np.asarray(dataset1[0]))

    train_x, train_y, test_x, test_y = cvt.split_train_test(0.75, example)
    total_len = len(train_x)
    total_len -= total_len % (batch_size * timesteps)
    train_x = train_x[:total_len]
    train_y = train_y[:total_len]

    # sample_weights = map(dict_mapper(prior), train_x)
    # sample_weights = np.reshape(list(sample_weights), (math.floor(total_len/timesteps), timesteps))

    train_x = np.reshape(train_x, (math.floor(total_len/timesteps), timesteps, label_card))
    train_y = np.reshape(train_y, (math.floor(total_len/timesteps), timesteps, label_card))
    print(train_x.shape)
    validation_ratio = None

    if validation:
        validation_len = (total_len * 0.1)
        validation_ratio = (validation_len - (validation_len % (batch_size * timesteps))) / total_len

    total_len = len(test_x)
    total_len -= total_len % (batch_size * timesteps)
    test_x = test_x[:total_len]
    test_y = test_y[:total_len]
    test_x = np.reshape(test_x, (math.floor(total_len/timesteps), timesteps, label_card))
    test_y = np.reshape(test_y, (math.floor(total_len/timesteps), timesteps, label_card))

    wcc = weighted_categorical_crossentropy(inverse)

    model = create_model(cell_size, (train_x.shape[1], train_x.shape[2]), stateful=True,
                         batch=batch_size,
                         output_dim=label_card * prediction_len,
                         layers=layers,
                         loss=wcc)
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

    return scores, confusion, normal, model


if __name__ == "__main__":



    layer_set = set([1])
    cell_set = set([512])
    offset_set = set([0])

    fileNames = glob.glob('./data/*.data')

    for filename in fileNames:
        for layer in layer_set:
            for cell in cell_set:
                for offset in offset_set:
                    info = run_instance(filename, cell_size=cell, layers=layer, epoch=1000, batch_size=1000, prediction_len=1, offset=0, timesteps=100)
                    if info is None:
                        continue
                    normal_mat = info[2]
                    scores = info[0]
                    print(normal_mat)
                    print(scores)