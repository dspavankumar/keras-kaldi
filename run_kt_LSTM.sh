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


set -e

nj=4
. cmd.sh
. path.sh

## Configurable directories
train=data/train
test=data/test
lang=data/lang
gmm=exp/tri2b
exp=exp/dnn_5b

## Split training data into train and cross-validation sets
[ -d ${train}_tr95 ] || utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 $train ${train}_tr95 ${train}_cv05

## Align data using GMM
for dset in cv05 tr95; do
    [ -f ${gmm}_ali_$dset/ali.1.gz ] || steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    ${train}_$dset $lang $gmm ${gmm}_ali_$dset
done

## Train
[ -f $exp/dnn.nnet.h5 ] || python3 steps_kt/train_LSTM.py ${train}_cv05 ${gmm}_ali_cv05 ${train}_tr95 ${gmm}_ali_tr95 $gmm $exp

## Make graph
[ -f $gmm/graph/HCLG.fst ] || utils/mkgraph.sh ${lang}_test_bg $gmm $gmm/graph

## Decode
[ -f $exp/decode/wer_11 ] || bash steps_kt/decode_seq.sh --nj $nj \
    --add-deltas "true" --norm-vars "true" --splice-size "11" \
    $test $gmm/graph $exp $exp/decode

#### Align
##    [ -f ${exp}_ali ] || steps_kt/align_seq.sh --nj $nj --cmd "$train_cmd" \
##        --add-deltas "true" --norm-vars "true" --splice-opts "11" \
##        $train $lang $exp ${exp}_ali

