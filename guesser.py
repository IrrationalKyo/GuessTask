from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np
import math

import converter as cvt

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


def create_model(cell_count, shape, stateful, batch, output_dim):
    model = Sequential()
    model.add(LSTM(cell_count,
              input_shape=shape,
			  batch_size=batch,
              stateful=stateful,
			  return_sequences=True))
    # model.add(Dense(cell_count, activation='relu'))
    # # model.add(Dropout(0.3))
    # model.add(LSTM(cell_count,
    #                input_shape=(cell_count,output_dim),
    #                batch_size=batch,
    #                stateful=stateful,
    #                return_sequences=True))
    model.add(Dense(output_dim, activation='relu'))
    model.compile(loss='poisson', optimizer='adam', metrics=['accuracy'])
    return model

def run(filename, fold_count, n, label_card, factor, validation_frac):
    # raw_data = cvt.text_to_list(filename)

    # (train_x, train_y), (test_x,test_y) = cvt.list_to_array(raw_data[:9999], 0.8, 100, 16)

    # folds = cvt.generate_folds(fold_count, raw_data[:math.floor(len(raw_data)/factor)], validation_frac, n, label_card)
    pure_raw_data = cvt.text_to_list(filename)
    raw_data = cvt.list_to_example(pure_raw_data,label_card,n)
    folds = cvt.generate_folds2(5,raw_data)
    print(folds[0][1][0].shape)
    fold_scores = []

    for i in range(fold_count):
        model = create_model(50, (n, label_card), True, 1, label_card)
        train = folds[i]
        fit_history = model.fit(train[0], train[1], epochs=5, batch_size=1, verbose=1)

        score_list = []
        for  j in range(fold_count):
            if i == j:
                continue
            model.reset_states()
            scores = model.evaluate(folds[j][0], folds[j][1], batch_size=1, verbose=0)
            score_list.append(scores)
        fold_scores.append(score_list)

    score_display(fold_scores)

dataset = {'dataset_det_1.txt', 'dataset_det_1.txt'}
set_n ={50,75,100,150,200,300}


run('dataset_new_det_1.txt',5,1,16,2,0)
