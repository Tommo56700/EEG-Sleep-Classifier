import numpy as np
from tensorflow import keras as keras

class DataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, listIDs, labels, batchSize=32, dim=6000, numChannels=12, numClasses=5, shuffle=False):
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
    X = np.empty((self.batchSize, self.dim, self.numChannels))
    y = np.empty((self.batchSize), dtype = int)
    
    # Generate data
    for i, ID in enumerate(listIDsTemp):
      # Store sample
      X[i,] = np.load('/scratch/c.c1673374/data/' + ID + '.npy')
      
      # Store class
      y[i] = self.labels[ID]
      
    return X, keras.utils.to_categorical(y, num_classes = self.numClasses)