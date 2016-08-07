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
    
    spliceSize = 11
    if len(sys.argv) == 3:
        spliceSize = int(sys.argv[2])

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    ## Load model
    dnn = keras.models.load_model (model)

    arkIn = sys.stdin.buffer
    arkOut = sys.stdout.buffer
    encoding = sys.stdout.encoding
    signal (SIGPIPE, SIG_DFL)

    uttId, featMat = kaldiIO.readUtterance(arkIn)
    m, n = featMat.shape
    p, q = featMat.strides
    featMat = numpy.lib.stride_tricks.as_strided(featMat, strides=(p, p, q), shape=(m-spliceSize+1, spliceSize, n))

    while uttId:
        logProbMat = numpy.log (dnn.predict (featMat))
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
        uttId, featMat = kaldiIO.readUtterance(arkIn)
        m, n = featMat.shape
        p, q = featMat.strides
        featMat = numpy.lib.stride_tricks.as_strided(featMat, strides=(p, p, q), shape=(m-spliceSize+1, spliceSize, n))
   
