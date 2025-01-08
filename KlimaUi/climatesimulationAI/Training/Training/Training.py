import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM, MaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Conv1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import datetime
import math
from tensorflow.keras import layers


class AddPositionEmbs(layers.Layer):
    """inputs are image patches
    Custom layer to add positional embeddings to the inputs."""

    def __init__(self, posemb_init=None, **kwargs):
        super().__init__(**kwargs)
        self.posemb_init = posemb_init

    def get_config(self):
        config = super().get_config()
        config.update({
            "posemb_init": self.posemb_init,
        })
        return config

    def build(self, inputs_shape):
        pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
        self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

    def call(self, inputs, inputs_positions=None):
        # inputs.shape is (batch_size, seq_len, emb_dim).
        pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

        return inputs + pos_embedding


'''
part of ViT Implementation
this block implements the Transformer Encoder Block
Contains 3 parts--
1. LayerNorm 2. Multi-Layer Perceptron 3. Multi-Head Attention
For repeating the Transformer Encoder Block we use Encoder_f function. 
'''


def mlp_block_f(mlp_dim, inputs):
    x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
    x = layers.Dropout(rate=0.1)(x)  # dropout rate is from original paper,
    x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)  # check GELU paper
    x = layers.Dropout(rate=0.1)(x)
    return x


def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
    x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x)
    # self attention multi-head, dropout_rate is from original implementation
    x = layers.Add()([x, inputs])  # 1st residual part

    y = layers.LayerNormalization(dtype=x.dtype)(x)
    y = mlp_block_f(mlp_dim, y)
    y_1 = layers.Add()([y, x])  # 2nd residual part
    return y_1


def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
    initializer = tf.keras.initializers.GlorotNormal()
    x = AddPositionEmbs(posemb_init=initializer, name='posembed_input')(
        inputs)  # RandomNormal(stddev=0.02) #RandomUniform
    x = layers.Dropout(rate=0.2)(x)
    for _ in range(num_layers):
        x = Encoder1Dblock_f(num_heads, mlp_dim, x)

    encoded = layers.LayerNormalization(name='encoder_norm')(x)
    return encoded


def prepdata(features, targets, gafsize):
    # features: input parameters for the model, np.array
    # targets: training targets for the model, np.array
    # gafsize: size of the GAF matrices, int
    # preparation of training data for training (reshaping & test-train-split)
    X = np.array(features)
    X = X.reshape(features[:, 0, 0].size, gafsize, features[0, 0, :].size, 1)
    Y = np.array(targets)

    # test train split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1, shuffle=False)
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return (x_train, x_test, y_train, y_test)


def buildConv(filter_size=20, gafsize=25, inputshape=(100, 300, 1), ResNet=True, printsum=True):
    # filter_size: size of filters in all Convolutional Layers, int (only for ResNet)
    # gafsize: size of GAF matrices, int
    # input_shape: shape of the input for model, triple
    # ResNet: whether a ResNet architecture is to be used, boolean
    # printsum: whether a summary of the network archictecture is to be printed at the end, boolean

    # build a convolutional neural network (CNN)
    if ResNet:
        # build ResNet architecture
        inputs = keras.Input(shape=inputshape, name="img")  # input/block 1
        x = layers.Conv2D(64, filter_size, activation="relu")(inputs)
        block_1_output = x
        # block 2
        x = layers.Conv2D(64, filter_size, activation="relu", padding="same")(block_1_output)
        x = layers.Dropout(0.15)(x)
        block_2_output = layers.add([x, block_1_output])  # skip-connection
        # block 3
        x = layers.Conv2D(64, filter_size, activation="relu", padding="same")(block_2_output)
        x = layers.Dropout(0.15)(x)
        block_3_output = layers.add([x, block_2_output])  # skip-connection
        # block 4
        x = layers.Conv2D(64, filter_size, activation="relu", padding="same")(block_3_output)
        x = layers.Dropout(0.15)(x)
        block_4_output = layers.add([x, block_3_output])  # skip-connection
        # final CNN block with flattening
        x = layers.Conv2D(64, filter_size, activation="relu")(block_4_output)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.15)(x)
        # Dense Layers before output
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(1)(x)  # output/final layer

        # compiling the model with MSE loss and Adam optimizer
        model = keras.Model(inputs, outputs, name="kalaResNet")
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'accuracy'])

        if printsum:
            # print summary of the model
            model.summary()
    else:
        # build CNN with linear architecture and increasing filter size and dropout
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding='same', input_shape=inputshape))
        model.add(Dropout(0.15))
        model.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu', padding='same'))
        model.add(Dropout(0.15))
        model.add(Conv2D(filters=256, kernel_size=(4, 4), activation='relu', padding='same'))
        model.add(Dropout(0.15))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dropout(0.15))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1))
        # compiling the model with MSE loss and Adam optimizer
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'accuracy'])
        if printsum:
            # print summary of the model
            model.summary()
    return model


def buildSST(gafsize, outputsize, printsum, modelname, features, num_heads=8, old=False, new=False, sea=False,
             mapshape=(192, 288, 1), seasea=False):
    # gafsize: size of the GAF matrices, int
    # outputsize: amount of output targets, int
    # printsum: whether to print a summary of the model at the end, boolean
    # modelname: filename the model is to be saved under, string
    # features: input parameters, np.array
    # num_head: number of attention heads in the attention layers, int
    # mapshape: shape of the temperature grid (input for the sea level model)

    # build a model with SST/mViT architecture
    # there are 5 different architectures to be chosen from: old, new, sea, seasea and "normal"
    # the architectures can be enabled by setting the respective parameter to true
    # the "normal" architecture is enabled by setting all other architectures to false
    # (or set none of them, because they set false by default)
    # sea: for the sea level models considering regionalised temperature, e.g. kalaSST104

    if seasea:
        # #####################################
        # hyperparameter section
        # DO NOT RANDOMLY MESS WITH THESE HYPERPARAMETERS, THEY NEED TO BE IN A CERTAIN RATIO!!!
        # THESE HYPERPARAMETERS WHERE CALCULATED CONDITIONAL TO THE NEEDED RATIO!!!
        # See Appendix B and/or other.hyperparameters.findpossiblehyperparameters() to calculate new hyperparameter sets!

        # length and width of the input matrix (combined GAF)
        x = 20
        y = 140
        # transformer hyper parameters
        transformer_layers = 4
        patch_size = 10
        hidden_size = 625
        mlp_dim = 512
        n = 28  # number of filters in the convolutional layers
        f = 4  # filter size in the convolutional layers
        m = 6  # number of convolutional layers + 1
        ###############
        if ((x // patch_size) * (y // patch_size) * hidden_size) == (
                (x - ((m - 1) * (f - 1))) * (y - ((m - 1) * (f - 1))) * n):
            # input (global emission and temperature branch)
            pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))
            inputs = layers.Input(shape=(gafsize, features[0][0, 0, :].size, 1), name='gaf')
            # Convolution Block
            conv1 = layers.Conv2D(n, f, activation='relu')(inputs)
            conv2 = layers.Conv2D(n, f, activation='relu')(conv1)
            conv3 = layers.Conv2D(n, f, activation='relu')(conv2)
            conv4 = layers.Conv2D(n, f, activation='relu')(conv3)
            conv5 = layers.Conv2D(n, f, activation='relu')(conv4)
            block_in = layers.Dropout(0.05)(conv5)
            # LSTM Block
            reshape = tf.reshape(block_in, [-1, 125 * 28, 5])
            lstm = layers.LSTM(5, return_sequences=True)(
                reshape)
            lstm2 = layers.LSTM(5, return_sequences=True)(lstm)
            flatten = layers.Flatten()(lstm2)
            # transformer blocks
            row_axis, col_axis = (1, 2)  # channels last images
            seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
            x = tf.reshape(flatten, [-1, seq_len, hidden_size])
            encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, x)
            # final part global emissions and temperature branch
            im_representation = tf.reduce_mean(encoder_out, axis=1)
            lpout1 = layers.Dense(256)(im_representation)

            # regionalised temperature branch with ConvolutionLSTM Layers
            mapped_input = layers.Input(shape=mapshape, name='map')
            convlstm1_map = layers.Conv2D(n, f, activation='relu')(mapped_input)
            drop1_map = layers.Dropout(0.05)(convlstm1_map)
            convlstm2_map = layers.Conv2D(n, f, activation='relu')(drop1_map)
            convlstm3_map = layers.Conv2D(n, f, activation='relu')(convlstm2_map)
            convlstm4_map = layers.Conv2D(n, f, activation='relu')(convlstm3_map)
            convlstm5_map = layers.Conv2D(n, f, activation='relu')(convlstm4_map)
            drop2_map = layers.Dropout(0.05)(convlstm5_map)
            convlstm6_map = layers.Conv2D(n, f, activation='relu')(drop2_map)
            convlstm7_map = layers.Conv2D(n, f, activation='relu')(convlstm6_map)
            convlstm8_map = layers.Conv2D(n, f, activation='relu')(convlstm7_map)
            flatten_map = layers.Flatten()(convlstm8_map)

            # combining both branches
            x = layers.concatenate([lpout1, flatten_map])
            # final Dense
            lpout2 = layers.Dense(512)(x)
            logits = layers.Dense(outputsize)(lpout2)

            # compiling model
            model = tf.keras.Model(inputs=[inputs, mapped_input], outputs=logits, name=modelname)
            if printsum: model.summary()
            learning_rate = 8 * (10 ** -5)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                 epsilon=1e-9)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae', 'mape', 'accuracy'])

            return model

    if sea:
        # #####################################
        # hyperparameter section
        # DO NOT RANDOMLY MESS WITH THESE HYPERPARAMETERS, THEY NEED TO BE IN A CERTAIN RATIO!!!
        # THESE HYPERPARAMETERS WHERE CALCULATED CONDITIONAL TO THE NEEDED RATIO!!!
        # See Appendix B and/or other.hyperparameters.findpossiblehyperparameters() to calculate new hyperparameter sets!

        # length and width of the input matrix (combined GAF)
        x = 20
        y = 140
        # transformer hyperparameters
        transformer_layers = 4
        patch_size = 10
        hidden_size = 625
        mlp_dim = 512
        n = 28  # number of filters in the convolutional layers
        f = 4  # filter size in the convolutional layers
        m = 6  # number of convolutional layers + 1
        # #####################################
        if ((x // patch_size) * (y // patch_size) * hidden_size) == (
                (x - ((m - 1) * (f - 1))) * (y - ((m - 1) * (f - 1))) * n):
            # input (global emission and temperature branch)
            pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))
            inputs = layers.Input(shape=(gafsize, features[0][0, 0, :].size, 1), name='gaf')
            # Convolution Block
            conv1 = layers.Conv2D(n, f, activation='relu')(inputs)
            conv2 = layers.Conv2D(n, f, activation='relu')(conv1)
            conv3 = layers.Conv2D(n, f, activation='relu')(conv2)
            conv4 = layers.Conv2D(n, f, activation='relu')(conv3)
            conv5 = layers.Conv2D(n, f, activation='relu')(conv4)
            block_in = layers.Dropout(0.05)(conv5)
            # LSTM Block
            reshape = tf.reshape(block_in, [-1, 125 * 28, 5])
            lstm = layers.LSTM(5, return_sequences=True)(reshape)
            lstm2 = layers.LSTM(5, return_sequences=True)(lstm)
            flatten = layers.Flatten()(lstm2)
            # Transformer blocks
            row_axis, col_axis = (1, 2)  # channels last images
            seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
            x = tf.reshape(flatten, [-1, seq_len, hidden_size])
            encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, x)
            # final part global emissions and temperature branch
            im_representation = tf.reduce_mean(encoder_out, axis=1)
            lpout1 = layers.Dense(256)(im_representation)

            # regionalised temperature branch with ConvolutionLSTM Layers
            mapped_input = layers.Input(shape=mapshape, name='map')
            convlstm1_map = layers.Conv2D(n, f, activation='relu')(mapped_input)
            drop1_map = layers.Dropout(0.05)(convlstm1_map)
            convlstm2_map = layers.Conv2D(n, f, activation='relu')(drop1_map)
            convlstm3_map = layers.Conv2D(n, f, activation='relu')(convlstm2_map)
            convlstm4_map = layers.Conv2D(n, f, activation='relu')(convlstm3_map)
            convlstm5_map = layers.Conv2D(n, f, activation='relu')(convlstm4_map)
            drop2_map = layers.Dropout(0.05)(convlstm5_map)
            convlstm6_map = layers.Conv2D(n, f, activation='relu')(drop2_map)
            convlstm7_map = layers.Conv2D(n, f, activation='relu')(convlstm6_map)
            convlstm8_map = layers.Conv2D(n, f, activation='relu')(convlstm7_map)
            flatten_map = layers.Flatten()(convlstm8_map)

            # combining both branches
            x = layers.concatenate([lpout1, flatten_map])

            # final Dense
            lpout2 = layers.Dense(512)(x)
            logits = layers.Dense(outputsize)(lpout2)

            # compiling model
            model = tf.keras.Model(inputs=[inputs, mapped_input], outputs=logits, name=modelname)
            if printsum: model.summary()
            learning_rate = 8 * (10 ** -5)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                 epsilon=1e-9)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae', 'mape', 'accuracy'])

            return model
        else:
            raise Exception("Hyperparameters aren't in the right ratio!")
    else:
        if new:
            # #####################################
            # hyperparameter section
            # DO NOT RANDOMLY MESS WITH THESE HYPERPARAMETERS, THEY NEED TO BE IN A CERTAIN RATIO!!!
            # THESE HYPERPARAMETERS WHERE CALCULATED CONDITIONAL TO THE NEEDED RATIO!!!
            # See Appendix B and/or other.hyperparameters.findpossiblehyperparameters() to calculate new hyperparameter sets!
            # #####################################
            # length and width of the input matrix (combined GAF)
            x = 20
            y = 140
            # transformer hyperparameters
            transformer_layers = 4
            patch_size = 10
            hidden_size = 625
            mlp_dim = 512
            n = 28  # number of filters in Conv Layers
            f = 4  # filtersize in Conv Layers
            m = 6  # number of convolutional layers - 1
            ###############
            if ((x // patch_size) * (y // patch_size) * hidden_size) == (
                    (x - ((m - 1) * (f - 1))) * (y - ((m - 1) * (f - 1))) * n):
                # input
                pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))
                inputs = layers.Input(shape=(gafsize, features[0, 0, :].size, 1))
                # Convolution Block
                conv1 = layers.Conv2D(n, f, activation='relu')(inputs)
                conv2 = layers.Conv2D(n, f, activation='relu')(conv1)
                conv3 = layers.Conv2D(n, f, activation='relu')(conv2)
                conv4 = layers.Conv2D(n, f, activation='relu')(conv3)
                conv5 = layers.Conv2D(n, f, activation='relu')(conv4)
                block_in = layers.Dropout(0.05)(conv5)
                # LSTM Block
                reshape = tf.reshape(block_in, [-1, 125 * 28, 5])
                lstm = layers.LSTM(5, return_sequences=True)(
                    reshape)
                lstm2 = layers.LSTM(5, return_sequences=True)(lstm)
                flatten = layers.Flatten()(lstm2)

                # Transformer Blocks
                row_axis, col_axis = (1, 2)  # channels last images
                seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
                x = tf.reshape(flatten, [-1, seq_len, hidden_size])
                encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, x)
                #  final part (mlp to classification)
                im_representation = tf.reduce_mean(encoder_out, axis=1)
                lpout1 = layers.Dense(256)(im_representation)
                lpout2 = layers.Dense(512)(lpout1)
                logits = layers.Dense(outputsize)(lpout2)
                # compiling the model
                model = tf.keras.Model(inputs=inputs, outputs=logits, name=modelname)
                if printsum: model.summary()
                learning_rate = 8 * (10 ** -5)
                optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                     epsilon=1e-9)
                model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae', 'mape', 'accuracy'])
                return model
            else:
                raise Exception("Hyperparameters aren't in the right ratio!")
        if not old:
            # #####################################
            # hyperparameter section
            # DO NOT RANDOMLY MESS WITH THESE HYPERPARAMETERS, THEY NEED TO BE IN A CERTAIN RATIO!!!
            # THESE HYPERPARAMETERS WHERE CALCULATED CONDITIONAL TO THE NEEDED RATIO!!!
            # See Appendix B and/or other.hyperparameters.findpossiblehyperparameters() to calculate new hyperparameter sets!
            # #####################################
            # length and width of the input matrix (combined GAF)
            x = 20
            y = 140
            # transformer hyperparameters
            transformer_layers = 2
            patch_size = 10
            hidden_size = 625
            mlp_dim = 512
            n = 28  # number of filters in Conv Layers
            f = 4  # filtersize in Conv Layers
            m = 6  # number of convolutional layers - 1
            ###############
            if ((x // patch_size) * (y // patch_size) * hidden_size) == (
                    (x - ((m - 1) * (f - 1))) * (y - ((m - 1) * (f - 1))) * n):
                # input
                pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))
                inputs = layers.Input(shape=(gafsize, features[0, 0, :].size, 1))
                # Convolution Block
                conv1 = layers.Conv2D(n, f, activation='relu')(inputs)
                conv2 = layers.Conv2D(n, f, activation='relu')(conv1)
                conv3 = layers.Conv2D(n, f, activation='relu')(conv2)
                conv4 = layers.Conv2D(n, f, activation='relu')(conv3)
                conv5 = layers.Conv2D(n, f, activation='relu')(conv4)
                block_in = layers.Dropout(0.05)(conv5)
                # LSTM Block
                reshape = tf.reshape(block_in, [-1, 125 * 28, 5])
                lstm = layers.LSTM(5, return_sequences=True)(
                    reshape)
                flatten = layers.Flatten()(lstm)
                # Transformer Blocks
                row_axis, col_axis = (1, 2)  # channels last images
                seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
                x = tf.reshape(flatten, [-1, seq_len, hidden_size])
                encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, x)
                #  final part (mlp to classification)
                im_representation = tf.reduce_mean(encoder_out, axis=1)
                lpout1 = layers.Dense(256)(im_representation)
                lpout2 = layers.Dense(512)(lpout1)
                logits = layers.Dense(outputsize)(lpout2)

                # compiling the model
                model = tf.keras.Model(inputs=inputs, outputs=logits, name=modelname)
                if printsum == True: model.summary()
                learning_rate = 8 * (10 ** -5)
                optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                     epsilon=1e-9)
                model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae', 'mape', 'accuracy'])
                return model
            else:
                raise Exception("Hyperparameters aren't in the right ratio!")
        else:
            # #####################################
            # hyperparameter section
            # DO NOT RANDOMLY MESS WITH THESE HYPERPARAMETERS, THEY NEED TO BE IN A CERTAIN RATIO!!!
            # THESE HYPERPARAMETERS WHERE CALCULATED CONDITIONAL TO THE NEEDED RATIO!!!
            # See Appendix B and/or other.hyperparameters.findpossiblehyperparameters() to calculate new hyperparameter sets!
            # #####################################
            # length and width of the input matrix (combined GAF)
            x = 20
            y = 140
            # transformer hyperparameters
            transformer_layers = 2
            patch_size = 10
            hidden_size = 625
            num_heads = 4
            mlp_dim = 512
            n = 28  # number of filters in Conv Layers
            f = 4  # filtersize in Conv Layers
            m = 6  # number of convolutional layers - 1
            ###############
            if ((x // patch_size) * (y // patch_size) * hidden_size) == (
                    (x - ((m - 1) * (f - 1))) * (y - ((m - 1) * (f - 1))) * n):
                # input
                pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))
                inputs = layers.Input(shape=(gafsize, features[0, 0, :].size, 1))
                # Convolution Block
                conv1 = layers.Conv2D(n, f, activation='relu')(inputs)
                conv2 = layers.Conv2D(n, f, activation='relu')(conv1)
                conv3 = layers.Conv2D(n, f, activation='relu')(conv2)
                conv4 = layers.Conv2D(n, f, activation='relu')(conv3)
                conv5 = layers.Conv2D(n, f, activation='relu')(conv4)
                block_in = layers.Dropout(0.05)(conv5)
                # LSTM Block
                reshape = tf.reshape(block_in, [-1, 125 * 28, 5])  # (None, 5, 125, 28)
                lstm = layers.LSTM(5, return_sequences=True)(
                    reshape)
                flatten = layers.Flatten()(lstm)
                # Transformer Blocks
                row_axis, col_axis = (1, 2)  # channels last images
                seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
                x = tf.reshape(flatten, [-1, seq_len, hidden_size])
                encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, x)
                #  final part (mlp to classification)
                im_representation = tf.reduce_mean(encoder_out, axis=1)
                lpout1 = layers.Dense(256)(im_representation)
                lpout2 = layers.Dense(512)(lpout1)
                logits = layers.Dense(outputsize)(lpout2)

                # compiling the model
                model = tf.keras.Model(inputs=inputs, outputs=logits, name=modelname)
                if printsum == True: model.summary()
                learning_rate = 8 * (10 ** -5)
                optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                     epsilon=1e-9)
                model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae', 'mape', 'accuracy'])
                return model
            else:
                raise Exception("Hyperparameters aren't in the right ratio!")


class training:
    # class for training of the models
    def __init__(self):
        pass

    def train(self, modelname: str, features, targets, gafsize, epochs, init=0, printsum=True, sst=True, conv=False,
              ResNet=True, valx=None, valy=None, continuetrain=False, num_heads=8, old=False, new=False):
        # modelname: filename the model is to be saved under, has to end with '.h5', string
        # features: input parameters (X training data), np.array
        # targets: target/outparameters (Y training data), np.array
        # gafsize: size of the GAF matrices, int
        # epochs: number of epoch the model is to be trained, int
        # init: start epoch (when continuing training), int
        # printsum: whether a sum of the model is to printed after building it, boolean
        # sst: whether to use a SST/mViT architecture, boolean
        # conv: whether to use an only CNN architecture, boolean
        # ResNet: whether to use a ResNet architecture (only for conv=true), boolean
        # valx: separate validation feature dataset, np.array, shape matching to features
        # valy: separate validation target dataset, np.array, shape matching to targets
        # continuetrain: whether to continue the training of a model or start a new one, boolean
        # num_heads: number of attention heads in the attention layers (only for sst=true)
        # old: use "old" architecture, boolean
        # new: use "new" architecture, boolean

        # preparation of training data
        x_train, x_test, y_train, y_test = prepdata(features, targets, gafsize)
        try:
            if valy == None:
                valy = y_test
                valx = x_test
        except:
            pass

        # build model with chosen architecture
        if sst:
            if not old:
                model = buildSST(gafsize, y_train[0, :].size, printsum, modelname, features, num_heads=num_heads,
                                 new=new)
            else:
                model = buildSST(gafsize, y_train[0, :].size, printsum, modelname, features, num_heads=num_heads,
                                 old=True, new=new)
        elif conv:
            model = buildConv(gafsize=gafsize, inputshape=x_train[0].shape, ResNet=ResNet)
        #init variables
        numepochs = epochs
        starttime = datetime.datetime.now()
        if continuetrain:
            # load weights as trained before if to continue a previous training
            model.load_weights('../climatesimulationAI/models/' + modelname)
        # set saving check points with given modelname
        mc = ModelCheckpoint('models/' + modelname, monitor='loss', mode='auto', verbose=1, save_best_only=False)
        # train the model
        model_history = model.fit(x_train, y_train, validation_data=(valx, valy), epochs=numepochs, initial_epoch=init,
                                  batch_size=20, verbose=1, shuffle=False, callbacks=[mc])
        # print time stats after the training
        print('start time: ' + str(starttime) + ' end time: ' + str(datetime.datetime.now()) + ' (' + str(
            numepochs) + ' Epochen Training ' + str(modelname) + ' )')

        return model_history, model, x_train, y_train

    def plotloss(self, model_history, m):
        # model_history: model.fit() return object from training a model
        # m: epoch to start plotting with
        # plots the loss and validation loss of a model
        plt.plot(model_history.history['loss'][m:])
        plt.plot(model_history.history['val_loss'][m:])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def plotcomppred(self, model, n, X, Y):
        # model: trained model object
        # n: index (geo) of results to be plotted, int
        # X: input (training) data for the model, np.array, shape according to the model's input shape
        # Y: target data to compare with, shape according to the model's output shape
        # plots prediction vs. targets of a trained model
        predictions = model.predict(X)
        plt.figure(figsize=(20, 5))
        plt.plot(Y[:, n])
        plt.plot(predictions[:, n])
        plt.title("Actual vs MAE Predictions")
        plt.xlabel("num")
        plt.ylabel("value")
        plt.legend(['Actual', 'Predictions'])
