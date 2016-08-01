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


import numpy
import struct

## Read utterance
def readUtterance (ark):
    ## Read utterance ID
    uttId = b''
    c = ark.read(1)
    if not c:
        return None, None
    while c != b' ':
        uttId += c
        c = ark.read(1)
    ## Read feature matrix
    header = struct.unpack('<xcccc', ark.read(5))
    m, rows = struct.unpack('<bi', ark.read(5))
    n, cols = struct.unpack('<bi', ark.read(5))
    featMat = numpy.frombuffer(ark.read(rows * cols * 4), dtype=numpy.float32)
    return uttId.decode(), featMat.reshape((rows,cols))

def writeUtterance (uttId, featMat, ark, encoding):
    featMat = numpy.asarray (featMat, dtype=numpy.float32)
    m,n = featMat.shape
    ## Write header
    ark.write (struct.pack('<%ds'%(len(uttId)),uttId.encode(encoding)))
    ark.write (struct.pack('<cxcccc',' '.encode(encoding),'B'.encode(encoding),
                'F'.encode(encoding),'M'.encode(encoding),' '.encode(encoding)))
    ark.write (struct.pack('<bi', 4, m))
    ark.write (struct.pack('<bi', 4, n))
    ## Write feature matrix
    ark.write (featMat)

