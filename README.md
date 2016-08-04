# Keras Interface for Kaldi ASR

## Why these Routines?

This code interfaces Kaldi tools for Speech Recognition and Keras 
tools for Deep Learning. Keras simplifies the latest deep 
learning implementations, unifies the two popular Theano and 
Tensorflow libraries, and has a growing user base. Kaldi, one of 
the best tools for ASR, thus needs an interface with Keras tools, 
and here is one. This code directly interacts with Kaldi style 
directories of data and alignments to build and test Deep 
Learning models in Keras.

## Features

1. Trains DNNs from Kaldi GMM system

2. Works with standard Kaldi data and alignment directories

3. Supports mini-batch training

4. Supports maxout and dropout training

5. Easily extendable to other deep learning implementations in 
  Keras

6. Decodes test utterances in Kaldi style

## Dependencies

1. Python 3.4+

2. Keras with Theano/Tensorflow backend

3. Kaldi

## Using the Code

Train a GMM system in Kaldi. Place steps_kt and run_kt.sh in the 
working directory. Configure and run run_kt.sh.

## Code Components

1. train.py is the Keras training script. DNN structure (type of 
  network, activations, number of hidden layers and nodes) can be 
  configured in this script.

2. dataGenerator.py provides an object that reads Kaldi data and 
  alignment directories in batches and retrieves mini-batches for 
  training.

3. nnet-forward.py passes test features through the trained DNNs 
  and outputs log probabilities (log of DNN outputs) in Kaldi 
  format.

4. kaldiIO.py reads and writes Kaldi-type binary features.

5. decode.py is the decoding script.

## Training Schedule

The script uses stochastic gradient descent with 0.5 momentum. It 
starts with a learning rate of 0.1 for a minimum of 10 
iterations. When the validation loss reduces by less than 0.002 
between successive iterations, learning rate is halved. Training 
is continued till learning rate is scaled 18 times.

## Results on Timit Phone Recognition

Timit database of 8 kHz sampling rate was used to train monophone,
triphone (300 pdfs), LDA+MLLT (500 pdfs) and DNN models.
Phone error rates are as follows:

1. Monophone: 34.25%

2. Triphone: 30.44%

3. LDA+MLLT: 27.03%

4. DNN: 23.70%

## Contributors
D S Pavan Kumar

dspavankumar [at] gmail [dot] com

## License
GNU GPL v3
