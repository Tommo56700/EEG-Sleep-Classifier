import pickle
import sys
import numpy as np
from tensorflow import keras as keras
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
  #testIDs = load_obj('testIDs')
  #tests = [testIDs[:1100], testIDs[1100:2100], testIDs[2100:3100], testIDs[3100:4100], testIDs[4100:5100], testIDs[5100:]]

  params = {'dim': INPUT_SHAPE[0],
            'batchSize': BATCH_SIZE,
            'numClasses': NUM_CLASSES,
            'shuffle': True}
    
  callbacksList = [keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25), keras.callbacks.ModelCheckpoint('E1/' + modelName + '_wts_b.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
    
  trainIDs = load_obj('trainIDs2')
  valIDs = load_obj('valIDs2')
  testIDs = load_obj('testIDs2')
  
  trainingGen = DataGeneratorSplit(trainIDs, labels)
  validationGen = DataGeneratorSplit(valIDs, labels)
  testingGen = DataGeneratorSplit(testIDs, labels)

  keras.backend.clear_session()
  
  model = Model.factory(modelName, INPUT_SHAPE, NUM_CLASSES)
  
  model.fit(BATCH_SIZE, EPOCHS, trainingGen, validationGen, callbacksList)

  model.load_weights('E1/' + modelName + '_wts_b.hdf5')

  model.test(testingGen)

  #y_pred = []
  #acc = 0

  #for test in tests:
    
    #batch_size = len(test)

    #params = {'dim': INPUT_SHAPE[0],
    #          'batchSize': batch_size,
    #          'numClasses': NUM_CLASSES,
    #          'shuffle': False}
    
    #predictGen = DataGeneratorSplit(test, labels, **params)
    #acc += model.test(predictGen)
    #y_pred.append(model.predict(predictGen))

  #p = np.concatenate(y_pred)
  #print("Total accuracy:", (acc/len(tests)))

  #save_obj(p, modelName + '_p')

if __name__ == '__main__':
  main(sys.argv[1])