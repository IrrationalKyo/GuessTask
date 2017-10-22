from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import converter as cvt

def create_model(cell_count, shape, stateful, batch, output_dim):
    model = Sequential()
    model.add(LSTM(cell_count,
              input_shape=shape,
			  batch_size=batch,
              stateful=stateful,
			  return_sequences=True))
    model.add(Dense(output_dim))
    model.compile(loss='squared_hinge', optimizer='adam', metrics=['accuracy'])
    return model


(train_x, train_y), (test_x,test_y) = cvt.list_to_array(cvt.text_to_list('dataset_1.txt'), 0.8, 1000, 16)
model = create_model(1000, (1, 1000), False, 1, 16)

print(train_x.shape)

model.fit(train_x, train_y, epochs=1, batch_size=1)

scores = model.evaluate(test_x, test_y)
print("Accuracy: %.2f%%" % (scores[1]*100))