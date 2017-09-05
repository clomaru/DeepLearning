import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

lrNum = 0.1
batch = 4

def plot_history(history):
    plt.plot(history.history['loss'],"o-",label="",)
    plt.title('XOR(sigmoid(lr='+str(lrNum)+'))')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

# Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# Model
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=lrNum),
              metrics=['accuracy'])

his = model.fit(X, Y, epochs=4000, batch_size=batch)

loss_and_metrics = model.evaluate(X,Y)


classes = model.predict_classes(X, batch_size=batch)
prob = model.predict_proba(X, batch_size=batch)

print('classified:\n{0}\n'.format(Y == classes))
print('output probability:\n{0}\n'.format(prob))
print(loss_and_metrics)
plot_history(his)
