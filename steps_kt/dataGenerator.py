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


from subprocess import Popen, PIPE, DEVNULL
import tempfile
import kaldiIO
import pickle
import struct
import numpy
import os

## Data generator class for Kaldi
class dataGenerator:
    def __init__ (self, data, ali, exp, batchSize=256):
        self.data = data
        self.ali = ali
        self.exp = exp
        self.batchSize = batchSize
        
        self.maxSplitDataSize = 100 ## These many utterances are loaded into memory at once.
        self.labelDir = tempfile.TemporaryDirectory()
        aliPdf = self.labelDir.name + '/alipdf.txt'
 
        ## Generate pdf indices
        Popen (['ali-to-pdf', exp + '/final.mdl',
                    'ark:gunzip -c %s/ali.*.gz |' % ali,
                    'ark,t:' + aliPdf]).communicate()

        ## Read labels
        with open (aliPdf) as f:
            labels, self.numFeats = self.readLabels (f)
       
        self.inputFeatDim = 429 ## IMPORTANT: HARDCODED. Change if necessary.
        self.outputFeatDim = self.readOutputFeatDim()
        self.splitDataCounter = 0
        
        self.x = numpy.empty ((0, self.inputFeatDim))
        self.y = numpy.empty ((0, self.outputFeatDim))
        self.batchPointer = 0
        self.doUpdateSplit = True

        ## Read number of utterances
        with open (data + '/utt2spk') as f:
            self.numUtterances = sum(1 for line in f)
        self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)

        ## Split data dir
        if not os.path.isdir (data + 'split' + str(self.numSplit)):
            Popen (['utils/split_data.sh', data, str(self.numSplit)]).communicate()
        
        ## Save split labels and delete labels
        self.splitSaveLabels (labels)

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()
        
    ## Determine the number of output labels
    def readOutputFeatDim (self):
        p1 = Popen (['gmm-info', '%s/final.mdl' % self.exp], stdout=PIPE)
        modelInfo = p1.stdout.read().splitlines()
        for line in modelInfo:
            if b'number of pdfs' in line:
                return int(line.split()[-1])

    ## Load labels into memory
    def readLabels (self, aliPdfFile):
        labels = {}
        numFeats = 0
        for line in aliPdfFile:
            line = line.split()
            numFeats += len(line)-1
            labels[line[0]] = [int(i) for i in line[1:]]
        return labels, numFeats
    
    ## Save split labels into disk
    def splitSaveLabels (self, labels):
        for sdc in range (1, self.numSplit+1):
            splitLabels = {}
            with open (self.data + '/split' + str(self.numSplit) + '/' + str(sdc) + '/utt2spk') as f:
                for line in f:
                    uid = line.split()[0]
                    if uid in labels:
                        splitLabels[uid] = labels[uid]
            with open (self.labelDir.name + '/' + str(sdc) + '.pickle', 'wb') as f:
                pickle.dump (splitLabels, f)

    ## Convert integer labels to binary
    def getBinaryLabels (self, intLabelList):
        numLabels = len(intLabelList)
        binaryLabels = numpy.zeros ((numLabels, self.outputFeatDim))
        binaryLabels [range(numLabels),intLabelList] = 1
        return binaryLabels
  
    ## Return a batch to work on
    def getNextSplitData (self):
        p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data + '/split' + str(self.numSplit) + '/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE, stderr=DEVNULL)
        p2 = Popen (['splice-feats','--print-args=false','--left-context=5','--right-context=5',
                'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()
        p3 = Popen (['add-deltas','--print-args=false','ark:-','ark:-'], stdin=p2.stdout, stdout=PIPE)
        p2.stdout.close()

        with open (self.labelDir.name + '/' + str(self.splitDataCounter) + '.pickle', 'rb') as f:
            labels = pickle.load (f)
    
        featList = []
        labelList = []
        while True:
            uid, featMat = kaldiIO.readUtterance (p3.stdout)
            if uid == None:
                return (numpy.vstack(featList), numpy.vstack(labelList))
            if uid in labels:
                labelMat = self.getBinaryLabels(labels[uid])
                featList.append (featMat)
                labelList.append (labelMat)

    ## Make the object iterable
    def __iter__ (self):
        return self

    ## Retrive a mini batch
    def __next__ (self):
        while (self.batchPointer + self.batchSize >= len (self.x)):
            if not self.doUpdateSplit:
                self.doUpdateSplit = True
                break

            self.splitDataCounter += 1
            x,y = self.getNextSplitData()
            self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate ((self.y[self.batchPointer:], y))
            self.batchPointer = 0

            if self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
                self.doUpdateSplit = False
        
        xMini = self.x[self.batchPointer:self.batchPointer+self.batchSize]
        yMini = self.y[self.batchPointer:self.batchPointer+self.batchSize]
        self.batchPointer += self.batchSize
        return (xMini, yMini)
