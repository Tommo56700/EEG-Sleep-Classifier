import numpy as np
from tensorflow import keras as keras

class DataGeneratorSplit(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, listIDs, labels, batchSize=32, dim=6000, numChannels=[10, 1, 1], numClasses=5, shuffle=False):
    'Initialization'
    self.dim = dim
    self.batchSize = batchSize
    self.labels = labels
    self.listIDs = listIDs
    self.numChannels = numChannels
    self.numClasses = numClasses
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.listIDs) / self.batchSize))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index * self.batchSize:(index + 1) * self.batchSize]
    
    # Find list of IDs
    listIDsTemp = [self.listIDs[k] for k in indexes]
    
    # Generate data
    X, y = self.__data_generation(listIDsTemp)
    
    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.listIDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, listIDsTemp):
    'Generates data containing batchSize samples' # X : (numSamples, dim, numChannels)
    # Initialization
    X = []
    for chans in self.numChannels:
      X.append(np.empty((self.batchSize, self.dim, chans)))
    y = np.empty((self.batchSize), dtype = int)
    
    # Generate data
    for i, ID in enumerate(listIDsTemp):
      # Store sample
      X[0][i,] = np.load('/scratch/c.c1673374/data/' + ID + '.npy')[:, :10]
      X[1][i,] = np.load('/scratch/c.c1673374/data/' + ID + '.npy')[:, 10].reshape((6000,1))
      X[2][i,] = np.load('/scratch/c.c1673374/data/' + ID + '.npy')[:, 11].reshape((6000,1))
      
      # Store class
      y[i] = self.labels[ID]
      
    return X, keras.utils.to_categorical(y, num_classes = self.numClasses)