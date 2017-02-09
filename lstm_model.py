img_rows, img_cols = 28, 28
img_channels = 1
nb_classes = 10
lstm_output_size = 70
batch_size = trainx.shape[0]
print(batch_size)
nb_epoch = 2

model = Sequential()

model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'),
                         batch_input_shape=np.append([batch_size,10],trainx.shape[1:])))
print(model.output_shape)
model.add(Activation('relu'))
model.add(TimeDistributed(Convolution2D(16, 2, 2)))
model.add(Activation('relu'))

model.add(TimeDistributed(Flatten()))

print(model.output_shape)

model.add(LSTM(24,
               input_shape=(testx.shape[0], 11664),
               return_sequences=False,
               stateful=True))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
