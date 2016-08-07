#!/usr/bin/python3

##  Copyright (C) 2016 D S Pavan Kumar
##  dspavankumar [at] gmail [dot] com
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.


import keras
from keras.optimizers import SGD
from dataGenerator import dataGenerator
import sys
import os

if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 7:
    raise TypeError ('USAGE: train.py data_cv ali_cv data_tr ali_tr gmm_dir dnn_dir')

data_cv = sys.argv[1]
ali_cv  = sys.argv[2]
data_tr = sys.argv[3]
ali_tr  = sys.argv[4]
gmm     = sys.argv[5]
exp     = sys.argv[6]

## Learning parameters
learning = {'rate' : 0.1,
            'batchSize' : 256,
            'minEpoch' : 10,
            'lrScale' : 0.5,
            'lrScaleCount' : 18,
            'minValError' : 0.002}

os.makedirs (exp, exist_ok=True)

trGen = dataGenerator (data_tr, ali_tr, gmm, learning['batchSize'])
cvGen = dataGenerator (data_cv, ali_cv, gmm, learning['batchSize'])

## Initialise learning parameters and models
s = SGD(lr=learning['rate'], decay=0, momentum=0.5, nesterov=True)
m = keras.models.Sequential([
                keras.layers.MaxoutDense(1024, nb_feature=3, input_dim=trGen.inputFeatDim),
                keras.layers.Dropout(0.2),
                keras.layers.MaxoutDense(1024, nb_feature=3),
                keras.layers.Dropout(0.2),
                keras.layers.MaxoutDense(1024, nb_feature=3),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(trGen.outputFeatDim, activation='softmax')])

## Initial training
m.compile(loss='categorical_crossentropy', optimizer=s, metrics=['accuracy'])
print ('Learning rate: %f' % learning['rate'])
h = [m.fit_generator (trGen, samples_per_epoch=trGen.numFeats, 
        validation_data=cvGen, nb_val_samples=cvGen.numFeats,
        nb_epoch=learning['minEpoch'], verbose=1)]
m.save (exp + '/dnn.nnet.h5', overwrite=True)

## Refine learning based on validation loss
prevValError = h[-1].history['val_loss'][-2]
while True:
    valErrorDiff = prevValError - h[-1].history['val_loss'][-1]
    if learning['lrScaleCount']==0 and valErrorDiff<learning['minValError']:
        break
    elif valErrorDiff < learning['minValError']:
        learning['rate'] *= learning['lrScale']
        print ('Learning rate: %f' % learning['rate'])
        learning['lrScaleCount'] -= 1
        m.optimizer.lr.set_value(learning['rate'])
    
    h.append (m.fit_generator (trGen, samples_per_epoch=trGen.numFeats,
            validation_data=cvGen, nb_val_samples=cvGen.numFeats,
            nb_epoch=1, verbose=1))
    prevValError = h[-2].history['val_loss'][-1]
    m.save (exp + '/dnn.nnet.h5', overwrite=True)

