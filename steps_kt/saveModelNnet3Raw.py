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


## NOTE: This script has limited functionality. It currently converts
##       feedforward networks with relu and softmax layers in HDF5 format
##       to the standard Kaldi's nnet3 "raw" format. Call this script from
##       steps_kt/saveModelNnet3.sh to get a complete model.

import keras
import numpy
import sys

def saveModel (model, fileName):
    with open (fileName, 'w') as f:
        f.write ('<Nnet3> \n')

        ## Write the component descriptions
        f.write ('input-node name=input dim=%d\n' % m.input_shape[-1])
        prevLayerName = 'input'
        num_components = 0
        for layer in model.layers:
            if layer.name.startswith ('dense'):
                f.write ('component-node name=%s.affine component=%s.affine input=%s\n' % (layer.name, layer.name, prevLayerName))
                num_components += 1
                activation_text = layer.get_config()['activation']
                if activation_text != 'linear':
                    f.write ('component-node name=%s.%s component=%s.%s input=%s.affine\n' % (layer.name, activation_text, layer.name, activation_text, layer.name))
                    num_components += 1
                prevLayerName = layer.name + '.' + activation_text
        f.write('output-node name=output input=%s objective=linear\n' % prevLayerName)

        f.write('\n<NumComponents> %d\n' % num_components)
        
        ## Write the layer values
        for layer in model.layers:
            if not layer.name.startswith ('dense'):
                raise TypeError ('Unknown layer type: ' + layer.name)
            
            f.write ('<ComponentName> %s.affine <NaturalGradientAffineComponent> <MaxChange> 2.0 <LearningRate> 0.001 <LinearParams>  [ \n ' % (layer.name))
            for row in layer.get_weights()[0].T:
                row.tofile (f, format="%e", sep=' ')
                f.write (' \n ')
            f.write ('] \n <BiasParams> [ ')
            layer.get_weights()[1].tofile (f, format="%e", sep=' ')
            f.write (' ] \n')
            f.write ('<RankIn> 20 <RankOut> 80 <UpdatePeriod> 4 <NumSamplesHistory> 2000 <Alpha> 4 <IsGradient> F </NaturalGradientAffineComponent>\n')

            ## Deal with the activation
            activation_text = layer.get_config()['activation']
            if activation_text == 'relu':
                f.write ('<ComponentName> %s.relu <RectifiedLinearComponent> <Dim> %d <ValueAvg> [ ] <DerivAvg> [ ] <Count> 0 <NumDimsSelfRepaired> 0 <NumDimsProcessed> 0 </RectifiedLinearComponent>\n' % (layer.name, layer.output_shape[-1]))
            elif activation_text == 'softmax':
                f.write ('<ComponentName> %s.softmax <LogSoftmaxComponent> <Dim> %d <ValueAvg> [ ] <DerivAvg> [ ] <Count> 0 <NumDimsSelfRepaired> 0 <NumDimsProcessed> 0 </LogSoftmaxComponent>\n' % (layer.name, layer.output_shape[-1]))
            else:
                raise TypeError ('Unknown/unhandled activation: ' + activation_text)
        f.write ('</Nnet3> \n')

## Save h5 model in nnet3 format
if __name__ == '__main__':
    h5model = sys.argv[1]
    nnet3 = sys.argv[2]
    m = keras.models.load_model (h5model)
    saveModel(m, nnet3)
