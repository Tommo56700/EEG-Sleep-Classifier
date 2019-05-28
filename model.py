import time
import numpy as np
from tensorflow import keras as keras
from utils import *

class Model(object):
  def factory(type, inputShape, numClasses):
    if type == "cnn": return CNN(inputShape, numClasses)
    if type == "cnn2": return CNN2(inputShape, numClasses)
    if type == "multi": return CNN_Multi(inputShape, numClasses)
    if type == "multi2": return CNN_Multi2(inputShape, numClasses)
    if type == "lstm": return LSTM(inputShape, numClasses)
    if type == "para": return CNNLSTM(inputShape, numClasses)
    if type == "res": return ResNet(inputShape, numClasses)
    if type == "spec": return CNN_Spec(inputShape, numClasses)

    raise AssertionError("Bad model creation: " + type)

  def fit(self, batchSize, numEpochs, trainingGen, validationGen, callbackList):
    startTime = time.time()
    
    hist = self.model.fit_generator(generator = trainingGen,
                                    steps_per_epoch = trainingGen.__len__(),
                                    epochs = numEpochs,
                                    verbose = 1,
                                    callbacks = callbackList,
                                    validation_data = validationGen,
                                    validation_steps = validationGen.__len__(),
                                    use_multiprocessing = True,
                                    workers = 6)
    duration = time.time() - startTime
    print("Training time:", duration)

  def load_weights(self, path):
    self.model.load_weights(path)

  def test(self, testingGen):
    score = self.model.evaluate_generator(generator = testingGen, use_multiprocessing = True, workers = 6, verbose = 1)
    
    print("\nAccuracy on test data: %0.4f" % score[1])
    print("\nLoss on test data: %0.4f" % score[0])

    return score[1]

  def predict(self, testingGen, y_test):
    y_predict = self.model.predict_generator(generator = testingGen, use_multiprocessing = False, verbose = 1)
    confusion_matrix(y_test, y_predict)

  def predict_single(self, x_test, y_test):
    y_predict = self.model.predict(x_test)
    
    plot_sleep_histogram(y_test, y_predict)
    plot_confusion_matrix(np.argmax(y_test, axis = 1), np.argmax(y_predict, axis = 1))

class CNN(Model):
  def __init__(self, inputShape, numClasses):
    self.model = self.build(inputShape, numClasses)
    self.model.summary()
    
  def build(self, inputShape, numClasses):
    inputLayer = keras.layers.Input(inputShape)
    
    conv1 = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputLayer)
    conv1 = keras.layers.MaxPooling1D(pool_size = 5)(conv1)
    
    conv2 = keras.layers.Conv1D(filters = 48, kernel_size = 5, padding = 'valid', activation = 'relu')(conv1)
    conv2 = keras.layers.GlobalAveragePooling1D()(conv2)
    
    denseLayer = keras.layers.Dense(40, activation = 'relu')(conv2)
    denseLayer = keras.layers.Dropout(0.1)(denseLayer)
    
    outputLayer = keras.layers.Dense(numClasses, activation = 'softmax')(denseLayer)
    
    modelCNN = keras.models.Model(inputs = inputLayer, outputs = outputLayer)
    modelCNN.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
    
    return modelCNN

class CNN2(Model):
  def __init__(self, inputShape, numClasses):
    self.model = self.build(inputShape, numClasses)
    self.model.summary()
    
  def build(self, inputShape, numClasses):
    inputLayer = keras.layers.Input(inputShape)
    
    conv1 = keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding="valid")(inputLayer)
    conv1 = keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding="valid")(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
    conv1 = keras.layers.SpatialDropout1D(rate=0.01)(conv1)
    
    conv2 = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv1)
    conv2 = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
    conv2 = keras.layers.SpatialDropout1D(rate=0.01)(conv2)
    
    conv3 = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv2)
    conv3 = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv3)
    conv3 = keras.layers.MaxPooling1D(pool_size=2)(conv3)
    conv3 = keras.layers.SpatialDropout1D(rate=0.01)(conv3)
    
    #try higher channels
    conv4 = keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding="valid")(conv3)
    conv4 = keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding="valid")(conv4)
    conv4 = keras.layers.GlobalAveragePooling1D()(conv4)
    conv4 = keras.layers.Dropout(rate=0.01)(conv4)

    denseLayer = keras.layers.Dense(64, activation='relu')(conv4)
    denseLayer = keras.layers.Dropout(rate=0.05)(denseLayer)
    
    outputLayer = keras.layers.Dense(numClasses, activation = 'softmax')(denseLayer)
    
    modelCNN = keras.models.Model(inputs = inputLayer, outputs = outputLayer)
    modelCNN.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001), metrics = ['accuracy'])
    
    return modelCNN

class CNN_Multi(Model):
  def __init__(self, inputShape, numClasses):
    self.model = self.build(inputShape, numClasses)
    self.model.summary()
    
  def build(self, inputShape, numClasses):
    inputEEG = keras.layers.Input((6000, 10))
    inputEMG = keras.layers.Input((6000, 1))
    inputEOG = keras.layers.Input((6000, 1))

    x = keras.layers.Conv1D(filters = 16, kernel_size = 5, padding = 'valid', activation = 'relu')(inputEEG)
    x = keras.layers.Conv1D(filters = 16, kernel_size = 5, padding = 'valid', activation = 'relu')(x)
    x = keras.layers.MaxPooling1D(pool_size = 2)(x)
    
    x = keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = 'valid', activation = 'relu')(x)
    x = keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = 'valid', activation = 'relu')(x)
    x = keras.layers.MaxPooling1D(pool_size = 2)(x)
    
    x = keras.layers.Conv1D(64, kernel_size = 3, activation='relu', padding="valid")(x)
    x = keras.layers.Conv1D(64, kernel_size = 3, activation='relu', padding="valid")(x)
    x = keras.layers.MaxPooling1D(pool_size = 2)(x)
    
    x = keras.layers.Conv1D(64, kernel_size = 3, activation='relu', padding="valid")(x)
    x = keras.layers.Conv1D(64, kernel_size = 3, activation='relu', padding="valid")(x)
    x = keras.layers.AveragePooling1D()(x)    
    x = keras.layers.Flatten()(x)
    
    y = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputEMG)
    y = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(y)
    y = keras.layers.MaxPooling1D(pool_size = 2)(y)
    
    y = keras.layers.Conv1D(filters = 64, kernel_size = 3, padding = 'valid', activation = 'relu')(y)
    y = keras.layers.Conv1D(filters = 64, kernel_size = 3, padding = 'valid', activation = 'relu')(y)
    y = keras.layers.GlobalAveragePooling1D()(y)
    
    
    z = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputEOG)
    z = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(z)
    z = keras.layers.MaxPooling1D(pool_size = 2)(z)
    
    z = keras.layers.Conv1D(filters = 64, kernel_size = 3, padding = 'valid', activation = 'relu')(z)
    z = keras.layers.Conv1D(filters = 64, kernel_size = 3, padding = 'valid', activation = 'relu')(z)
    z = keras.layers.GlobalAveragePooling1D()(z)
    
    
    denseLayer = keras.layers.concatenate([x, y, z])
    denseLayer = keras.layers.Dense(500, activation = 'relu')(denseLayer)
    denseLayer = keras.layers.Dropout(0.2)(denseLayer)
    
    outputLayer = keras.layers.Dense(5, activation = 'softmax')(denseLayer)
    
    modelCNN = keras.models.Model(inputs = [inputEEG, inputEMG, inputEOG], outputs = outputLayer)
    modelCNN.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
    
    return modelCNN

class CNN_Multi2(Model):
  def __init__(self, inputShape, numClasses):
    self.model = self.build(inputShape, numClasses)
    self.model.summary()
    
  def build(self, inputShape, numClasses):
    inputEEG = keras.layers.Input((6000, 10))
    inputEEGShaped = keras.layers.Reshape((6000, 10, 1))(inputEEG)
    inputEMG = keras.layers.Input((6000, 1))
    inputEOG = keras.layers.Input((6000, 1))
    
    x = keras.layers.Conv2D(filters = 6, kernel_size = (1, 10), padding = 'valid', activation = 'linear')(inputEEGShaped)
    x = keras.layers.Reshape((6000, 6, 1))(x)

    x = keras.layers.Conv2D(filters = 16, kernel_size = (5, 1), padding = 'valid', activation = 'relu')(x)
    x = keras.layers.Conv2D(filters = 16, kernel_size = (5, 1), padding = 'valid', activation = 'relu')(x)
    x = keras.layers.MaxPooling2D(pool_size = (3, 1))(x)
    
    x = keras.layers.Conv2D(filters = 32, kernel_size = (3, 1), padding = 'valid', activation = 'relu')(x)
    x = keras.layers.Conv2D(filters = 32, kernel_size = (3, 1), padding = 'valid', activation = 'relu')(x)
    x = keras.layers.MaxPooling2D(pool_size = (3, 1))(x)
    
    x = keras.layers.Conv2D(64, kernel_size = (3, 1), activation='relu', padding="valid")(x)
    x = keras.layers.Conv2D(64, kernel_size = (3, 1), activation='relu', padding="valid")(x)
    x = keras.layers.MaxPooling2D(pool_size = (3, 1))(x)
    
    x = keras.layers.Conv2D(64, kernel_size = (3, 1), activation='relu', padding="valid")(x)
    x = keras.layers.Conv2D(64, kernel_size = (3, 1), activation='relu', padding="valid")(x)
    x = keras.layers.AveragePooling2D(pool_size = (216, 1))(x)
    x = keras.layers.Flatten()(x)    
    
    
    y = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputEMG)
    y = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(y)
    y = keras.layers.MaxPooling1D(pool_size = 2)(y)
    
    y = keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation = 'relu')(y)
    y = keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation = 'relu')(y)
    y = keras.layers.GlobalAveragePooling1D()(y)
    
    
    z = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputEOG)
    z = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(z)
    z = keras.layers.MaxPooling1D(pool_size = 2)(z)
    
    z = keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation = 'relu')(z)
    z = keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation = 'relu')(z)
    z = keras.layers.GlobalAveragePooling1D()(z)
    
    
    denseLayer = keras.layers.concatenate([x, y, z])
    denseLayer = keras.layers.Dense(400, activation = 'relu')(denseLayer)
    denseLayer = keras.layers.Dropout(0.2)(denseLayer)
    
    outputLayer = keras.layers.Dense(5, activation = 'softmax')(denseLayer)
    
    modelCNN = keras.models.Model(inputs = [inputEEG, inputEMG, inputEOG], outputs = outputLayer)
    modelCNN.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
    
    return modelCNN

class CNN_Spec(Model):
  def __init__(self, inputShape, numClasses):
    self.model = self.build(inputShape, numClasses)
    self.model.summary()
    
  def build(self, inputShape, numClasses):
    inputEEG = keras.layers.Input((6000, 10))
    inputEMG = keras.layers.Input((6000, 1))
    inputEOG = keras.layers.Input((6000, 1))

    x1 = keras.layers.Conv1D(filters = 32, kernel_size = 13, padding = 'valid', activation = 'relu')(inputEEG)
    x1 = keras.layers.AveragePooling1D(pool_size = 5)(x1)
    
    x1 = keras.layers.Conv1D(filters = 48, kernel_size = 13, padding = 'valid', activation = 'relu')(x1)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)
    
    x2 = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputEEG)
    x2 = keras.layers.AveragePooling1D(pool_size = 3)(x2)
    
    x2 = keras.layers.Conv1D(filters = 48, kernel_size = 5, padding = 'valid', activation = 'relu')(x2)
    x2 = keras.layers.GlobalAveragePooling1D()(x2)
    
    x3 = keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = 'valid', activation = 'relu')(inputEEG)
    x3 = keras.layers.AveragePooling1D(pool_size = 2)(x3)
    
    x3 = keras.layers.Conv1D(filters = 48, kernel_size = 3, padding = 'valid', activation = 'relu')(x3)
    x3 = keras.layers.GlobalAveragePooling1D()(x3)
        
    
    y1 = keras.layers.Conv1D(filters = 32, kernel_size = 13, padding = 'valid', activation = 'relu')(inputEMG)
    y1 = keras.layers.AveragePooling1D(pool_size = 5)(y1)
    
    y1 = keras.layers.Conv1D(filters = 48, kernel_size = 13, padding = 'valid', activation = 'relu')(y1)
    y1 = keras.layers.GlobalAveragePooling1D()(y1)
    
    y2 = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputEMG)
    y2 = keras.layers.AveragePooling1D(pool_size = 3)(y2)
    
    y2 = keras.layers.Conv1D(filters = 48, kernel_size = 5, padding = 'valid', activation = 'relu')(y2)
    y2 = keras.layers.GlobalAveragePooling1D()(y2)
    
    y3 = keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = 'valid', activation = 'relu')(inputEMG)
    y3 = keras.layers.AveragePooling1D(pool_size = 2)(y3)
    
    y3 = keras.layers.Conv1D(filters = 48, kernel_size = 3, padding = 'valid', activation = 'relu')(y3)
    y3 = keras.layers.GlobalAveragePooling1D()(y3)
      
    
    z1 = keras.layers.Conv1D(filters = 32, kernel_size = 13, padding = 'valid', activation = 'relu')(inputEOG)
    z1 = keras.layers.AveragePooling1D(pool_size = 5)(z1)
    
    z1 = keras.layers.Conv1D(filters = 48, kernel_size = 13, padding = 'valid', activation = 'relu')(z1)
    z1 = keras.layers.GlobalAveragePooling1D()(z1)
    
    z2 = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputEOG)
    z2 = keras.layers.AveragePooling1D(pool_size = 3)(z2)
    
    z2 = keras.layers.Conv1D(filters = 48, kernel_size = 5, padding = 'valid', activation = 'relu')(z2)
    z2 = keras.layers.GlobalAveragePooling1D()(z2)
    
    z3 = keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = 'valid', activation = 'relu')(inputEOG)
    z3 = keras.layers.AveragePooling1D(pool_size = 2)(z3)
    
    z3 = keras.layers.Conv1D(filters = 48, kernel_size = 3, padding = 'valid', activation = 'relu')(z3)
    z3 = keras.layers.GlobalAveragePooling1D()(z3)
        
    
    concatLayer = keras.layers.concatenate([x1, x2, x3, y1, y2, y3, z1, z2, z3])
    concatLayer = keras.layers.Dropout(0.1)(concatLayer)

    denseLayer = keras.layers.Dense(300, activation = 'relu')(concatLayer)
    denseLayer = keras.layers.Dropout(0.1)(denseLayer)

    denseLayer = keras.layers.Dense(50, activation = 'relu')(denseLayer)
    denseLayer = keras.layers.Dropout(0.1)(denseLayer)
    
    outputLayer = keras.layers.Dense(5, activation = 'softmax')(denseLayer)
    
    modelCNN = keras.models.Model(inputs = [inputEEG, inputEMG, inputEOG], outputs = outputLayer)
    modelCNN.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
    
    return modelCNN

class ResNet(Model):
  def __init__(self, inputShape, numClasses):
    self.model = self.build(inputShape, numClasses)
    self.model.summary()
    
  def build(self, inputShape, numClasses):
    input_layer = keras.layers.Input(inputShape)
    
    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu', padding='same')(input_layer)
    conv_y = keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(conv_x)
    conv_z = keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(conv_y)
    
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=32, kernel_size=1, padding='same')(input_layer)
    
    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    
    # BLOCK 2
    conv_x = keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu', padding='same')(output_block_1)
    conv_y = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(conv_x)
    conv_z = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv_y)
    
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=64, kernel_size=1, padding='same')(output_block_1)
    
    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    
    # BLOCK 3
    conv_x = keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu', padding='same')(output_block_2)
    conv_y = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(conv_x)
    conv_z = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv_y)
    
    # no need to expand channels because they are equal
    output_block_3 = keras.layers.add([output_block_2, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    
    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    
    output_layer = keras.layers.Dense(numClasses, activation='softmax')(gap_layer)
    
    modelRes = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    modelRes.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
    return modelRes

#class LSTM(Model):
#  def __init__(self, inputShape, numClasses):
#    self.model = self.build(inputShape, numClasses)
#    self.model.summary()
#    
#  def build(self, inputShape, numClasses):
#    inputLayer = keras.layers.Input(inputShape)
#    
#    conv1 = keras.layers.Conv1D(filters = 16, kernel_size = 5, padding = 'valid', activation = 'relu')(inputLayer)
#    conv1 = keras.layers.MaxPooling1D(pool_size = 5)(conv1)
#
#    x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(40))(conv1)
#    x = keras.layers.Dropout(0.5)(x)
#    
#    out = keras.layers.Dense(numClasses, activation='softmax')(x)
#    
#    modelCNN_LSTM = keras.models.Model(inputLayer, out)
#    modelCNN_LSTM.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001), metrics = ['accuracy'])
#    
#    return modelCNN_LSTM
#
#class LSTM1(Model):
#  def __init__(self, inputShape, numClasses):
#    self.model = self.build(inputShape, numClasses)
#    self.model.summary()
#    
#  def build(self, inputShape, numClasses):
#    inputLayer = keras.layers.Input(inputShape)
#    
#    conv1 = keras.layers.Conv1D(filters = 16, kernel_size = 5, padding = 'valid', activation = 'relu')(inputLayer)
#    conv1 = keras.layers.Conv1D(filters = 16, kernel_size = 5, padding = 'valid', activation = 'relu')(conv1)
#    conv1 = keras.layers.MaxPooling1D(pool_size = 5)(conv1)
#
#    x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(32))(conv1)
#    x = keras.layers.Dropout(0.5)(x)
#    
#    out = keras.layers.Dense(numClasses, activation='softmax')(x)
#    
#    modelCNN_LSTM = keras.models.Model(inputLayer, out)
#    modelCNN_LSTM.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001), metrics = ['accuracy'])
#    
#    return modelCNN_LSTM

class LSTM(Model):
  def __init__(self, inputShape, numClasses):
    self.model = self.build(inputShape, numClasses)
    self.model.summary()
        
  def build(self, inputShape, numClasses):
    inputLayer = keras.layers.Input(inputShape)
    
    conv1 = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid', activation = 'relu')(inputLayer)
    conv1 = keras.layers.MaxPooling1D(pool_size = 5)(conv1)

    conv2 = keras.layers.Conv1D(filters = 48, kernel_size = 5, padding = 'valid', activation = 'relu')(conv1)
    conv2 = keras.layers.MaxPooling1D(pool_size = 5)(conv2)

    x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(25))(conv2)
    x = keras.layers.Dropout(0.4)(x)
    
    out = keras.layers.Dense(numClasses, activation='softmax')(x)
    
    modelCNN_LSTM = keras.models.Model(inputLayer, out)
    modelCNN_LSTM.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001), metrics = ['accuracy'])
    
    return modelCNN_LSTM

#class CNNintoLSTM(Model):
#  def __init__(self, inputShape, numClasses):
#    self.model = self.build(inputShape, numClasses)
#    self.model.summary()
#    
#  def buildCNN(self, inputShape):
#    inputLayer = keras.layers.Input(inputShape)
#        
#    conv1 = keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding="valid")(inputLayer)
#    conv1 = keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding="valid")(conv1)
#    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
#    conv1 = keras.layers.SpatialDropout1D(rate=0.01)(conv1)
#    
#    conv2 = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv1)
#    conv2 = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv2)
#    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
#    conv2 = keras.layers.SpatialDropout1D(rate=0.01)(conv2)
#    
#    conv3 = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv2)
#    conv3 = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(conv3)
#    conv3 = keras.layers.MaxPooling1D(pool_size=2)(conv3)
#    conv3 = keras.layers.SpatialDropout1D(rate=0.01)(conv3)
#    
#    conv4 = keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding="valid")(conv3)
#    conv4 = keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding="valid")(conv4)
#    conv4 = keras.layers.GlobalAveragePooling1D()(conv4)
#    conv4 = keras.layers.Dropout(rate=0.01)(conv4)
#
#    denseLayer = keras.layers.Dense(64, activation='relu')(conv4)
#    denseLayer = keras.layers.Dropout(rate=0.05)(denseLayer)
#    
#    outputLayer = keras.layers.Dense(5, activation = 'softmax')(denseLayer)
#    
#    modelCNN = keras.models.Model(inputLayer, outputLayer)
#    #modelCNN.compile(keras.optimizers.Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['acc'])
#    
#    return modelCNN
#    
#  def build(self, inputShape, numClasses):   
#    inputLayer = keras.layers.Input(inputShape)
#    
#    cnn = self.buildCNN((6000, 12))
#    
#    conv = keras.layers.TimeDistributed(cnn)(inputLayer)
#                                       
#    lstm = keras.layers.CuDNNLSTM(8)(conv)
#
#    dense = keras.layers.Dense(8, activation="relu")(lstm)
#
#    out = keras.layers.Dense(numClasses, activation="softmax")(dense)
#    
#    modelCNNLSTM = keras.models.Model(inputLayer, out)
#
#    modelCNNLSTM.compile(keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics=['acc'])
#
#    return modelCNNLSTM

class CNNLSTM(Model):
  def __init__(self, inputShape, numClasses):
    self.model = self.build(inputShape, numClasses)
    self.model.summary()
    
  def build(self, inputShape, numClasses):
    inputLayer = keras.layers.Input(inputShape)
    
    x = keras.layers.Permute((2, 1))(inputLayer)
    x = keras.layers.CuDNNLSTM(8)(x)
    x = keras.layers.Dropout(0.5)(x)
    
    y = keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding="valid")(inputLayer)
    y = keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding="valid")(y)
    y = keras.layers.MaxPooling1D(pool_size=2)(y)
    y = keras.layers.SpatialDropout1D(rate=0.01)(y)
    
    y = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(y)
    y = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(y)
    y = keras.layers.MaxPooling1D(pool_size=2)(y)
    y = keras.layers.SpatialDropout1D(rate=0.01)(y)
    
    y = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(y)
    y = keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding="valid")(y)
    y = keras.layers.MaxPooling1D(pool_size=2)(y)
    y = keras.layers.SpatialDropout1D(rate=0.01)(y)
    
    y = keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding="valid")(y)
    y = keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding="valid")(y)
    y = keras.layers.GlobalAveragePooling1D()(y)
    
    x = keras.layers.concatenate([x, y])
    
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    
    out = keras.layers.Dense(numClasses, activation='softmax')(x)
    
    modelCNN_LSTM = keras.models.Model(inputLayer, out)
    modelCNN_LSTM.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001), metrics = ['accuracy'])
    
    return modelCNN_LSTM