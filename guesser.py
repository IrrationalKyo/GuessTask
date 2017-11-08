from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math

import converter as cvt

def create_model(cell_count, shape, stateful, batch, output_dim):
    model = Sequential()
    model.add(LSTM(cell_count,
              input_shape=shape,
			  batch_size=batch,
              stateful=stateful,
			  return_sequences=True))
    model.add(Dense(cell_count, activation='relu'))
    model.add(Dense(output_dim, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

raw_data = cvt.text_to_list('dataset_1.txt')

(train_x, train_y), (test_x,test_y) = cvt.list_to_array(raw_data[:9999], 0.8, 100, 16)
model = create_model(100, (100, 16), True, 1, 16)

print(train_x.shape)

model.fit(train_x, train_y, epochs=1, batch_size=1)

scores = model.evaluate(test_x, test_y, batch_size=1)
print('')
print(scores)
print("Accuracy: %.2f%%" % (scores[1]*100))
