import converter as cvt
import guesser as gue

''' Hacky scripts to varify results'''





if __name__ == "__main__":
    data_list = cvt.text_to_list('dataset_det_1.txt')
    folds = cvt.generate_time_series_folds(5,
                                           cvt.list_to_example_overlap_100(data_list[:int(len(data_list)/2)], 16),
                                           batch_size=10)
    i = 1
    for fold in folds:
        x_train = fold[0][0]
        y_train = fold[0][1]
        x_test = fold[1][0]
        y_test = fold[1][1]

        model = gue.create_model(25, (100, 16), stateful=True, batch=10, output_dim=16)
        model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=1)
        model.save("lstm25_drop0.3_lstm25_t100_batch10_fold"+str(i)+".model")
        i += 1

        print(model.evaluate(x_test, y_test, batch_size=10))

        gue.manual_verification(model, (x_test, y_test), batch_size=10)



