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


import sys
import numpy
import keras
import kaldiIO
from signal import signal, SIGPIPE, SIG_DFL

if __name__ == '__main__':
    model = sys.argv[1]
    priors = sys.argv[2]
    
    spliceSize = 11
    if len(sys.argv) == 4:
        spliceSize = int(sys.argv[3])

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    context = (spliceSize - 1) // 2

    ## Load model
    dnn = keras.models.load_model (model)
    p = numpy.genfromtxt (priors, delimiter=',')
    p[p==0] = 1e-5 ## Deal with zero priors

    arkIn = sys.stdin.buffer
    arkOut = sys.stdout.buffer
    encoding = sys.stdout.encoding
    signal (SIGPIPE, SIG_DFL)

    ## Load a feature matrix (utterance)
    uttId, featMat = kaldiIO.readUtterance(arkIn)

    while uttId:
        m, n = featMat.shape
        p, q = featMat.strides
        featMat = numpy.lib.stride_tricks.as_strided(featMat, strides=(p, p, q), shape=(m-spliceSize+1, spliceSize, n))

        ## Compute log-probabilities
        logProbMat = numpy.log (dnn.predict (featMat) / p)
        logProbMat [logProbMat == -numpy.inf] = -100

        ## Repeat logProb vectors at ends to match the number of features
        logProbMat = numpy.concatenate([numpy.tile(logProbMat[0],(context,1)), logProbMat, numpy.tile(logProbMat[-1],(context,1))])

        ## Write utterance
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)

        ## Load another feature matrix (utterance)
        uttId, featMat = kaldiIO.readUtterance(arkIn)
  
