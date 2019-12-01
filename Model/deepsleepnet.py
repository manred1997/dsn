"""
@article{Supratak2017,
    title = {DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG},
    author = {Supratak, Akara and Dong, Hao and Wu, Chao and Guo, Yike},
    journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
    year = {2017},
    month = {Nov},
    volume = {25},
    number = {11},
    pages = {1998-2008},
    doi = {10.1109/TNSRE.2017.2721116},
    ISSN = {1534-4320},
}
"""
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation
from keras.layers import Reshape, LSTM, TimeDistributed, Bidirectional, BatchNormalization
from keras.layers import concatenate, add
from keras.regularizers import l2
from keras.optimizers import Adam


def featurenet(summary=True):
    input_signal = Input(shape=(30 * 100, 1), name='input_signal')

    ######### CNNs with small filter size at the first layer #########
    smallfilter = Conv1D(filters=64, kernel_size=int(100/2), strides=int(100/16), padding='same', kernel_regularizer=l2(0.001), name='sConv1')(input_signal)
    smallfilter = BatchNormalization()(smallfilter)
    smallfilter = Activation(activation='relu')(smallfilter)
    smallfilter = MaxPool1D(pool_size=8, strides=8, name='sMaxP1')(smallfilter)
    smallfilter = Dropout(0.5, name='sDrop1')(smallfilter)

    smallfilter = Conv1D(filters=128, kernel_size=8, strides=1, padding='same', name='sConv2')(smallfilter)
    smallfilter = BatchNormalization()(smallfilter)
    smallfilter = Activation(activation='relu')(smallfilter)

    smallfilter = Conv1D( filters=128, kernel_size=8, strides=1, padding='same', name='sConv3')(smallfilter)
    smallfilter = BatchNormalization()(smallfilter)
    smallfilter = Activation(activation='relu')(smallfilter)

    smallfilter = Conv1D(kernel_size=8, filters=129, strides=1, padding='same', name='sConv4')(smallfilter)
    smallfilter = BatchNormalization()(smallfilter)
    smallfilter = Activation(activation='relu')(smallfilter)

    smallfilter = MaxPool1D(pool_size=4, strides=4, name='sMaxP2')(smallfilter)

    smallfilter = Reshape((int(smallfilter.shape[1]) * int(smallfilter.shape[2]),))(smallfilter) #Flatten

    ######### CNNs with large filter size at the first layer #########

    largefilter = Conv1D(filters=64, kernel_size=int(100*4), strides=int(100/2), padding='same', kernel_regularizer=l2(0.001), name='lConv1')(input_signal)
    largefilter = BatchNormalization()(largefilter)
    largefilter = Activation(activation='relu')(largefilter)
    largefilter = MaxPool1D(pool_size=4, strides=4, name='lMaxP1')(largefilter)
    largefilter = Dropout(0.5, name='lDrop1')(largefilter)

    largefilter = Conv1D(filters=128, kernel_size=6, strides=1, padding='same', name='lConv2')(largefilter)
    largefilter = BatchNormalization()(largefilter)
    largefilter = Activation(activation='relu')(largefilter)

    largefilter = Conv1D(filters=128, kernel_size=6, strides=1, padding='same', name='lConv3')(largefilter)
    largefilter = BatchNormalization()(largefilter)
    largefilter = Activation(activation='relu')(largefilter)

    largefilter = Conv1D(filters=128,kernel_size=6, strides=1, padding='same', name='lConv4')(largefilter)
    largefilter = BatchNormalization()(largefilter)
    largefilter = Activation(activation='relu')(largefilter)

    largefilter = MaxPool1D(pool_size=2, strides=2, name='lMaxP2')(largefilter)

    largefilter = Reshape((int(largefilter.shape[1]) * int(largefilter.shape[2]),))(largefilter)

    merged = concatenate([smallfilter, largefilter], name='concate')

    merged = Dense(1024)(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(5, name='merged')(merged)
    # print('merged',merged.shape)
    pre_softmax = Activation(activation='softmax')(merged)
    # print('pre_softmax',pre_softmax.shape)

    pre_model = Model(input_signal, pre_softmax)
    pre_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['acc'])
    if summary:
        pre_model.summary()
    return pre_model

def deepsleepnet(pre_model, summary=True):
    input_signal = pre_model.get_layer(name='input_signal').input
    merged = pre_model.get_layer(name='merged').output
    cnn_part = Model(input_signal, merged) # pre train 된 부분

    input_seq = Input(shape=(None, 3000, 1)) # sequence 3-dimension
    # print('input_seq', input_seq.shape)

    signal_sequence = TimeDistributed(cnn_part)(input_seq) # TimeDistributed 로 시퀀스를 입력 받을 수 있음
    # print('signal_sequence',signal_sequence.shape)

    bidirection = Bidirectional(LSTM(512, dropout=0.5, activation='relu', return_sequences=True),merge_mode='concat')(signal_sequence)
    # print('bidirection',bidirection.shape)

    fc1024 = Dense(1024)(signal_sequence)
    fc1024 = BatchNormalization()(fc1024)
    fc1024 = Activation(activation='relu')(fc1024)
    # print('fc1024',fc1024.shape)

    residual = add([bidirection, fc1024]) # skip-connection
    residual = Dropout(0.5)(residual)
    # print('residual',residual.shape)

    dense_seq = Dense(5)(residual)
    # print('dense_seq',dense_seq.shape)

    seq_softmax = Activation(activation='softmax')(dense_seq)
    # print('seq_softmax',seq_softmax.shape)

    seq_model = Model(input_seq, seq_softmax)
    seq_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-6), metrics=['acc'])
    if summary:
        seq_model.summary()
    return seq_model

if __name__=='__main__':
    pre_model = featurenet()
    seq_model = deepsleepnet(pre_model)
