import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# tensorboard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

# lossの履歴をプロット
def plot_history(history):
    plt.plot(history.history['loss'],label="MNIST LSTM",)
    plt.title('LSTM')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

log_filepath = './log' # tensorboard
###############################
#         データの生成          #
###############################
np.random.seed(0)
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 10000
indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

# 正規化
X = X / 255.0
X = X - X.mean(axis=1).reshape(len(X), 1)
X = X.reshape(len(X), 28, 28)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

###############################
#         モデルの設定          #
###############################
n_in = 28
n_time = 28
n_hidden = 128
n_out = 10

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

old_session = KTF.get_session() # tensorboard

with tf.Graph().as_default():
    # tensorboard
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)

    # image処理
    # 読み込むべきはX_trainではない気がする
    X_train_img = tf.reshape(X_train, [-1, 28, 28, 1])
    X_train_img = tf.cast(X_train_img, tf.float32)
    tf.summary.image('train',X_train_img , 10)

    model = Sequential()
    model.add(Bidirectional(LSTM(n_hidden),input_shape=(n_time, n_in)))
    model.add(Dense(n_out, kernel_initializer=weight_variable))
    model.add(Activation('softmax'))
    model.summary()

    # AdaDelta良い説！！！！！！！！！！！！１
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    tb_cb = keras.callbacks.TensorBoard(
                                log_dir=log_filepath,
                                histogram_freq=1,
                                embeddings_freq=100)
                                 # tensorboard

    ###############################
    #         モデルの学習          #
    ###############################
    # epochs = 60
    epochs = 3
    batch_size = 200

    his = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test,Y_test),
                    callbacks=[early_stopping,tb_cb]) # tensorboard

    ###############################
    #         モデルの予測          #
    ###############################
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('loss:{}'.format(score[0]))
    print('acc:{}'.format(score[1]))
    plot_history(his)

# tensorboard
KTF.set_session(old_session)
