import numpy as np
import keras
from keras import backend as K
from sklearn.metrics import confusion_matrix
import converter as cvt
import guesser as gue
import glob
import os

def makeFilenames(data_name, cell_size, epoch, batch_size, timesteps, offset):
    label_name = gue.tasksize_extractor(data_name)
    rep_number = gue.rep_extractor(data_name)

    result_path = "./result/" + str(label_name) + "/" + str(rep_number)
    id = "c"+ str(cell_size) + "_e" + str(epoch) + "_b" + str(batch_size) + "_ti" + str(timesteps) + "_o" + str(offset)+"_lstm"
    modelname = result_path + id + ".model"
    statname = result_path + id + ".json"

    return result_path, modelname, statname

def manualVerification(model, X, Y, batchSize):
    model.reset_states()
    predictions = model.predict(X, batch_size=batchSize)

    pred_Y = []

    total_len = len(predictions)

    for i in range(len(predictions)):
        pred_Y.append(np.argmax(predictions[i][0]))

    correct = 0


    print("prediction: " + str(pred_Y))
    Y = np.reshape(np.argmax(Y,axis=2),len(Y))
    print("acutal: " + str(Y))


    confusion = confusion_matrix(Y, pred_Y, labels=range(label_card))
    for i in range(label_card):
        correct += confusion[i][i]
    acc = float(correct / len(predictions))

    print("accuracy: " + str(acc))
    print(confusion)
    return confusion, acc

def manualConfidentVerification(model, X, Y, confidence, batchSize):
    model.reset_states()
    predictions = model.predict(X, batch_size=batchSize)

    predicted_Y = []
    corresponding_Y = []

    confidentCount = 0

    for i in range(len(predictions)):
        candidateIndex = np.argmax(predictions[i])
        if predictions[i][candidateIndex] > confidence:
            confidentCount += 1
            predicted_Y.append(candidateIndex)
            corresponding_Y.append(np.argmax(Y[i]))

    correct = 0

    confusion = confusion_matrix(corresponding_Y, predicted_Y, labels=range(label_card))
    for i in range(label_card):
        correct += confusion[i][i]
    acc = float(correct / len(predicted_Y))
    print("confident fraction: " + str(float(confidentCount)/len(predictions)))
    print("accuracy: " + str(acc))
    print(confusion)
    return confusion, acc


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

def vectorVisualizationCheck():

    return

# Many Multivariate  timesteps to Single Univariate
# regression to sequence output
# mmsu = (input) many multivariate (output)single univariate sequence
# input is of size (batch, timesteps, label_card * 2)
# output is of size (batch, 1, label_card)

if __name__ == "__main__":

    trainSplit = 0.8


    epoch = 50
    batchSize = 400
    timesteps = 30
    offset = 10000

    fileNames = glob.glob('./data/*.data')

    for fileName in fileNames:

        label_card = gue.tasksize_extractor(fileName)+1
        cell_size = 64

        modelExists = False

        result_path, modelname, statname = makeFilenames(fileName, cell_size, epoch, batchSize, timesteps, offset)
        originalTrace= cvt.readTraceFile(fileName)
        originalTrace = originalTrace[100:]
        # originalTrace = cvt.cut_random(originalTrace, 0.1, 0.01, fileName)
        # trim the first couple of sequence of original trace
        _, inverse = get_priors(originalTrace)

        if gue.file_exists(modelname):
            modelExists = True

        train_X = []
        train_Y = []
        test_X = []
        test_Y = []

        for t in range(label_card):
            print("processing task " + str(t))
            binaryTrace = gue.mask_trace(t, originalTrace)
            # trace = gue.uniform_sample_trace(5, trace)
            # trace = trace[100:1000000]

            regressionTrace, trace_Y = gue.regression_trace(binaryTrace)
            binaryTrace = binaryTrace[:len(regressionTrace)]
            binaryTrace = np.reshape(binaryTrace, (len(binaryTrace), 1))
            trace_X = [regressionTrace, binaryTrace]
            trace_X = np.stack(trace_X, axis=-1)  #<---- ISSUE WITH FORMATTING



            example = cvt.organizeTraceManyMultSingUni(
                trace_X, binaryTrace, timesteps, offset + timesteps)

            taskTrain_X, taskTrain_Y, taskTest_X, taskTest_Y = cvt.split_train_test(trainSplit, example)
            train_X.append(taskTrain_X)
            train_Y.append(taskTrain_Y)
            test_X.append(taskTest_X)
            test_Y.append(taskTest_Y)

        train_X = np.stack(train_X, axis=-1)
        train_Y = np.stack(train_Y, axis=-1)
        test_X = np.stack(test_X, axis=-1)
        test_Y = np.stack(test_Y, axis=-1)

        total_len = len(train_X)
        total_len -= total_len % batchSize
        train_X = train_X[:total_len]
        train_Y = train_Y[:total_len]

        train_X = np.reshape(train_X, (total_len, timesteps, label_card*2))
        train_Y = np.reshape(train_Y, (total_len, label_card))

        print(train_X[0])
        print(train_Y[0])


        validation_len = (total_len * 0.1)
        validation_ratio = (validation_len - (validation_len % (batchSize * timesteps))) / total_len

        total_len = len(test_X)
        total_len -= total_len % batchSize
        test_X = test_X[:total_len]
        test_Y = test_Y[:total_len]

        test_X = np.reshape(test_X, (total_len, timesteps, label_card*2))
        test_Y = np.reshape(test_Y, (total_len, label_card))
        wcc = weighted_categorical_crossentropy(inverse)

        #####
        if modelExists:
            model = keras.models.load_model(modelname, custom_objects={"loss":wcc})
            y_pred, y_true = manualConfidentVerification(model, test_X, test_Y, 0.99, batchSize)
            # y_pred, y_true = manualVerification(model, test_X, test_Y, batchSize)


        else:
            model = gue.create_model_many_sing_seq_lstm(cell_size, (timesteps, label_card*2), False,
                                 batchSize,
                                 output_dim=label_card,
                                 timesteps=timesteps,
                                 loss=wcc,
                                 optimizer="adam",
                                 layers=label_card-3)
            model.summary()

            for i in range(1):
                model.fit(train_X, train_Y, epochs=epoch, batch_size=batchSize,
                          verbose=2, validation_split=validation_ratio)
                model.reset_states()
                print("Current epoch: " + str(i))

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # TODO:
            model.save(modelname)

            scores = model.evaluate(test_X, test_Y, batch_size=batchSize, verbose=2)
            model.reset_states()
            y_pred, y_true = manualConfidentVerification(model, test_X, test_Y, 0.99, batchSize)
