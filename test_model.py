import pickle
import sys
import numpy as np
from tensorflow import keras as keras
from dataGenerator import DataGenerator
from dataGeneratorSplit import DataGeneratorSplit
from model import Model

def save_obj(obj, name):
    with open('E1/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
  with open('/scratch/c.c1673374/obj/' + name + '.pkl', 'rb') as f:
    return pickle.load(f)

def main(modelName):
  INPUT_SHAPE = (6000, 12)
  NUM_CLASSES = 5
  BATCH_SIZE = 16
  EPOCHS = 80

  labels = load_obj('sampleLabels')

  params = {'dim': INPUT_SHAPE[0],
            'batchSize': BATCH_SIZE,
            'numClasses': NUM_CLASSES,
            'numChannels': INPUT_SHAPE[1],
            'shuffle': True}
    
  callbacksList = [keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25), keras.callbacks.ModelCheckpoint('E1/' + modelName + '_wts_b.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
    
  trainIDs = load_obj('trainIDs2')
  valIDs = load_obj('valIDs2')
  testIDs = load_obj('testIDs2')
    
  trainingGen = DataGenerator(trainIDs, labels, **params)
  validationGen = DataGenerator(valIDs, labels, **params)
  testingGen = DataGenerator(testIDs, labels, **params)
    
  keras.backend.clear_session()
  
  model = Model.factory(modelName, INPUT_SHAPE, NUM_CLASSES)
  
  model.fit(BATCH_SIZE, EPOCHS, trainingGen, validationGen, callbacksList)

  model.load_weights('E1/' + modelName + '_wts_b.hdf5')

  model.test(testingGen)

if __name__ == '__main__':
  main(sys.argv[1])