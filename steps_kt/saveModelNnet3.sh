#!/bin/bash

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


## NOTE: This script converts feedforward DNNs in HDF5 format to the
##       standard Kaldi's nnet3 format. It has limited functionality.
##       It uses steps_kt/saveModelNnet3Raw.py, which is also limited
##       in functionality. The scripts do the job, but can be better.

exp=$1

. cmd.sh
. path.sh

## Check if argument exists
[ -z $exp ] && echo "Provide DNN directory as an argument" && exit 1

## Check if files required exist in the exp directory
for f in $exp/final.mdl $exp/dnn.nnet.h5 $exp/dnn.priors.csv ; do
    [ ! -f $f ] && echo "Expected $f to exist" && exit 1
done

## Copy the raw Nnet3
python3 scripts_kt/saveModelNnet3Raw.py $exp/dnn.nnet.h5 $exp/dnn.nnet3.raw

## Append context and priors to the raw Nnet3
printf "<LeftContext> 0 <RightContext> 0 <Priors>  [ " >> $exp/dnn.nnet3.raw
awk '{gsub(","," ",$0); print $0 " ]"}' $exp/dnn.priors.csv >> $exp/dnn.nnet3.raw

## Copy the transition matrix
copy-transition-model --binary=false $exp/final.mdl $exp/dnn.nnet3.trans

mv $exp/final.mdl $exp/final.mdl.bak

## Prepare the final model
cat $exp/dnn.nnet3.trans $exp/dnn.nnet3.raw > $exp/final.mdl.txt

## Convert to binary format
nnet3-am-copy $exp/final.mdl.txt $exp/final.mdl

## Clean up
rm -f $exp/dnn.nnet3.raw $exp/dnn.nnet3.trans $exp/final.mdl.txt

echo "Older final model backed up as: $exp/final.mdl.bak"
echo "Nnet3 model successfully stored as: $exp/final.mdl"
