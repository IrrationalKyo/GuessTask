import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import converter as cvt
from keras.models import load_model
import matplotlib.pyplot as plt

import time


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



def create_model(cell_count, shape, stateful, batch, output_dim, loss="poisson", drop_out = False):
    model = Sequential()
    model.add(LSTM(cell_count,
              input_shape=shape,
			  batch_size=batch,
              stateful=stateful,
			  return_sequences=True))
    model.add(Dense(cell_count, activation='relu'))
    if drop_out:
        model.add(Dropout(0.3))
    model.add(LSTM(cell_count,
                   input_shape=(cell_count,output_dim),
                   batch_size=batch,
                   stateful=stateful,
                   return_sequences=True))
    model.add(Dense(output_dim, activation='relu'))
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model
'''
    n = time steps
'''
def run(filename, fold_count, time_step, label_card, gap, batchSize):
    pure_raw_data = cvt.text_to_list(filename)
    raw_data = cvt.list_to_example_overlap(pure_raw_data, label_card)

    folds = cvt.generate_folds(5, (raw_data[0][:int(len(raw_data[0]) / 2)], raw_data[1][:int(len(raw_data[1]) / 2)]))

    fold_scores = []
    true_fold_scores = []

    for i in range(fold_count):
        model = create_model(100, (time_step, label_card), True, batch=batchSize, output_dim=label_card, loss="categorical_crossentropy")
        train = folds[i]
        fit_history = model.fit(train[0], train[1], epochs=1, batch_size=batchSize, verbose=1)
        model.save("model/foldIndex_" + str(i) + ".model")
        score_list = []
        true_score_list = []
        for  j in range(fold_count):
            if i == j:
                continue
            model.reset_states()
            # manually call predict
            scores = model.evaluate(folds[j][0], folds[j][1], batch_size=batchSize, verbose=1)
            confusion, trueAcc = manual_verification_100(model, folds[j], batch_size=batchSize)

            save_matrix(confusion, "train_"+str(i)+"_test_"+str(j)+".confusion")

            score_list.append(scores)
            true_score_list.append([None,trueAcc])
        fold_scores.append(score_list)
        true_fold_scores.append(true_score_list)

    score_display(fold_scores)
    score_display(true_fold_scores)

def manual_verification(model, test_dataset, batch_size=1):
    model.reset_states()
    y = model.predict(test_dataset[0], batch_size=batch_size)
    print(np.argmax(y,axis=2))
    confusion = confusion_matrix(np.argmax(test_dataset[1],axis=2), np.argmax(y,axis=2), labels=range(16))
    print(confusion)
    correct = 0
    for i in range(16):
        correct+=confusion[i][i]
    print("correct count: " + str(correct))
    print("acc: " + str(correct / len(y)))

    return (confusion, float(correct / len(y)))

def manual_verification_100(model, test_dataset, batch_size=1):
    model.reset_states()
    y = model.predict(test_dataset[0], batch_size=batch_size)
    # otuput shape will be the same as the input shape

    label_card = len(y[1][0])
    true_pred_y = []
    true_y = []
    for i in range(len(y)):
        pred_y = np.argmax(y[i][-1])
        true_pred_y.append(pred_y)
        label = np.argmax(test_dataset[1][i][-1])
        true_y.append(label)

    confusion = confusion_matrix(true_y, true_pred_y , labels=range(label_card))

    correct = 0
    for i in range(label_card):
        correct+=confusion[i][i]
    print("correct count: " + str(correct))
    print("acc: " + str(correct / len(y)))

    return (confusion, float(correct / len(y)))

def manual_verification_disjoint(model, test_dataset, batch_size=1):
    model.reset_states()
    y = model.predict(test_dataset[0], batch_size=batch_size)
    # otuput shape will be the same as the input shape
    time_step = len(y[1])
    print("time_step" + str(time_step))
    true_pred_y = []
    true_y = []
    for i in range(len(y)):
        for j in range(time_step):
            pred_y = np.argmax(y[i][j])
            true_pred_y.append(pred_y)
            label = np.argmax(test_dataset[1][i][j])
            true_y.append(label)

    confusion = confusion_matrix(true_y, true_pred_y , labels=range(16))

    correct = 0
    for i in range(16):
        correct+=confusion[i][i]
    print("correct count: " + str(correct))
    print("acc: " + str(correct / len(y)))

    return (confusion, float(correct / len(y)))

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

if __name__ == "__main__":
    # dataset = {'dataset_det_1.txt', 'dataset_det_1.txt'}
    # set_n ={50,75,100,150,200,300}

    run('dataset_new_det_1.txt', fold_count=5, time_step=100, label_card=16)

    # pure_raw_data = cvt.text_to_list('dataset_det_1.txt')
    # raw_data = cvt.list_to_example(pure_raw_data,16,1)
    # xy_train = cvt.chunk_examples(raw_data[0], raw_data[1], 0, 80000)
    # xy_test = cvt.chunk_examples(raw_data[0], raw_data[1], 80000, 99900)
    #
    # model = create_model(25, (1, 16), True, 1, 16)
    # model.fit(xy_train[0], xy_train[1], epochs=20, batch_size=1, verbose=1)
    # model.save("lstm_lstem_dataset_det1_" + str(int(time.time())) +".model")
    #
    # # model = load_model("lstm_lstem_dataset_det1_1512897026.model")
    # print(model.evaluate(xy_test[0], xy_test[1],batch_size=1))
    #
    # manual_verification(model, xy_test, 1)



    # print(result)