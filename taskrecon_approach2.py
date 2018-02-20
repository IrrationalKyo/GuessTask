from keras import backend as K
from keras.engine import Input, Model, InputSpec
from keras.layers import Dense, Activation, Dropout, Lambda
from keras.layers import Embedding, LSTM
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.models import Sequential

'''
    In this approach, I want to utilize the embedding representation of the tasks.
     

'''



INT_LSTM_HIDDEN_DIM = 50


def create_model(int_cell_count, int_input_length, int_label_card):
    input_layer = Input(shape=(int_input_length,), name="input_layer", dtype="int32")
    embedding_layer = Embedding(int_label_card, 32, input_length=int_input_length, trainable=True, mask_zero=False, name="embedding_layer")(input_layer)
    lstm_layer = LSTM(int_cell_count,
                    stateful=False,
                    return_sequences=True)(embedding_layer)
    vis_layer = Lambda(lambda x: x[:, -1, :], output_shape=(INT_LSTM_HIDDEN_DIM, ), name="vis_layer")(lstm_layer)
    output_layer = Dense(1,activation='softmax', name="output_layer")(vis_layer)
    model = Model(input=input_layer, output=output_layer)
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = create_model(100, 100, 16)