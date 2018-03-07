import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM
import numpy as np
from sklearn.metrics import confusion_matrix
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



def create_model(cell_count, shape, stateful, batch, output_dim, loss="categorical_crossentropy", drop_out = True):
    model = Sequential()
    model.add(CuDNNLSTM(cell_count,
              input_shape=shape,
			  batch_size=batch,
              stateful=stateful,
			  return_sequences=True))
    model.add(Dense(cell_count, activation='relu'))
    if drop_out:
        model.add(Dropout(0.5))

    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss=loss, optimizer='sgd', metrics=['accuracy'])
    return model


def run(filename, fold_count, time_step, label_card, gap, batchSize):
    pure_raw_data = cvt.text_to_list(filename)
    raw_data = cvt.list_to_example_sequence(pure_raw_data, label_card)

    folds = cvt.split_train_test(5, (raw_data[0][:int(len(raw_data[0]) / 2)], raw_data[1][:int(len(raw_data[1]) / 2)]))

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

    trace = cvt.newText_to_list("./data/size5rep0.data") [:100]
    for t in trace:
        if not isinstance(t, int):
            print("Not parsing correctly {}".format(t))
    example = cvt.list_to_example_sequence(trace, 16, offset=0, pred_len=1)
    for t in example[1]:
        if len(t) != 16:
            print("THIS OFFENDS ME {}".format(t))
    dataset1 = cvt.chunk_examples(example[0], example[1], 0, len(example[0]))
    print(np.asarray(dataset1[0]))
    train_x, train_y, test_x, test_y = cvt.split_train_test(0.75, example)
    total_len = len(train_x)
    total_len -= total_len % 32
    train_x = train_x[:total_len]
    train_y = train_y[:total_len]
    train_x = np.reshape(train_x,  (total_len, 1, len(train_x[0])))
    train_y = np.reshape(train_y,  (total_len, 1, len(train_y[0])))

    model = create_model(128, (train_x.shape[1],train_x.shape[2]), stateful=True,
                         batch=32,
                         output_dim=16)
    model.fit(train_x, train_y, epochs=20, batch_size=32, verbose=1)

