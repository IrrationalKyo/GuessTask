import converter as cvt
import guesser as gue
import os
import plotter
import matplotlib.pyplot as plt
from keras.models import load_model

''' load model and get utilfactor and normalized accuracy per dataset'''
def getAccuracyPerTask(datasetName, modelDir):
    model = load_model(modelDir)




if __name__ == "__main__":

    loss=["poisson","categorical_crossentropy"]
    node_size=[25,50,100,150]
    batch_size = [10,50,100]
    epoch = [10, 20]
    time_steps = [1, 10, 100]
    drop_out = [True, False]


    fold_count = 5

    data_list = cvt.text_to_list('dataset_new_det_1.txt')

    for l in loss:
        for n in node_size:
            for b in batch_size:
                for e in epoch:
                    for t in time_steps:
                        for d in drop_out:
                            n_size = n
                            b_size = b
                            time = t
                            ep = e
                            out = d
                            lo = l

                            id =  str(l) + "_" + str(n) + "_" + str(b) + "_" +  str(e) + "_" + str(t)+ "_" + str(out)
                            directory = "./" + id

                            folds = cvt.generate_time_series_folds(fold_count,
                                                                   cvt.list_to_example_overlap(data_list, 16),
                                                                   batch_size=b_size)
                            i = 1
                            for fold in folds:
                                x_train = fold[0][0]
                                y_train = fold[0][1]
                                x_test = fold[1][0]
                                y_test = fold[1][1]

                                file_name = "lstm_lstm_fold"+str(i)

                                model = gue.create_model(n_size, (time, 16), stateful=True, batch=b_size, output_dim=16, loss=l, drop_out=out)
                                model.fit(x_train, y_train, epochs=ep, batch_size=b_size, verbose=1)
                                if not os.path.exists(directory):
                                    os.makedirs(directory)
                                    os.makedirs(directory+"/normalized")
                                    os.makedirs(directory + "/unnormalized")
                                model.save(directory + "/" + file_name+".model")
                                i += 1

                                # print(model.evaluate(x_test, y_test, batch_size=batch_size))
                                (cnf_mat, acc) = gue.manual_verification_100(model, (x_test, y_test), batch_size=b_size)

                                plt.figure(figsize=(10, 10), dpi=100)
                                plotter.plot_confusion_matrix(cnf_mat, classes=range(16), normalize=True,
                                                      title='Normalized confusion matrix')

                                plt.savefig(directory + "/normalized/" + file_name + "_normalized_"+str(acc)+".png")
                                plt.figure(figsize=(10, 10), dpi=100)
                                plotter.plot_confusion_matrix(cnf_mat.astype(int), classes=range(16), normalize=False,
                                                      title='Non-Normalized confusion matrix')

                                plt.savefig(directory + "/unnormalized/" + file_name + "_"+str(acc)+".png")