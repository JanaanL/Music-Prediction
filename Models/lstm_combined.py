""" 
This module creates an LSTM model that uses a combined representation as input.
The input is a tuple consisting of a pitch and duration.  This tuple is then mapped
to an integer value using the dictionary that was created by calling the data.py file.
"""

import glob
import pickle
import numpy as np
import os
import math
import random
import utils
from music21 import converter, instrument, note, chord, stream, interval, pitch
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Input
from keras.layers import Embedding
from keras.models import Model
from keras.layers import Concatenate
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
import tensorflow as tf


def train_network():
    args = utils.get_args()
   
    #get dictionary of note/duration combinations
    with open(args.data_file + 'noteDurToInt', 'rb') as filepath:
        noteDurToInt = pickle.load(filepath)
    num_combinations = len(noteDurToInt)
    
    #Get data from midi files
    data = process_data(args, noteDurToInt)
    
    #Get model
    if args.checkpoint == None:
        model = create_model(args, num_combinations)
            
    #We are checkpointing
    else:
        model = load_model(args.checkpoint)

    train_model(model, data, args)
            
def process_data(args, noteDurToInt):
    
    # Create empty list for scores
    path = args.data_file + "trainList"
    with open(path, 'rb') as filepath:
        trainFileNames = pickle.load(filepath)

    scores = []
    for fname in trainFileNames:
        filePath = os.path.join(args.data_file + "prime_midi/",fname)
        try:
            midi_in = converter.parse(filePath)
            midi_out = converter.parse(os.path.join(args.data_file + "cont_true_midi/", fname))
            midi_in.append(midi_out)
            midi_in.flat

            if args.transpose == "all":
                transposed = utils.transposeToAll(midi_in)
                scores = scores + transposed
            
            if args.transpose == "C":
                midi_in = utils.transposeToC(midi_in)

            scores.append(midi_in)
        
        except:
            print("File would not parse correctly: ", fname)

    combinations = [[] for _ in scores]
  
    # Extract notes, chords, durations, and keys using Music21 Library
    print("Processing mono music files")
    for i, song in enumerate(scores):
        if i % 10 == 0:
            print ("Processing song no ", i)
        for element in song.flat:
            print(element)
            if isinstance(element, note.Note):
                combinations[i].append((element.pitch.midi, element.duration.quarterLength))
            elif isinstance(element, note.Rest):
                combinations[i].append((128, element.duration.quarterLength))

        print(combinations[i])
        combinations[i] = [noteDurToInt[n] for n in combinations[i]]
        print(combinations[i])

    sequenceLength = args.sequence_length
    train = []
    target = []
    num_combinations = len(noteDurToInt)
    
    # Construct training sequences for chords and durations
    for seq in combinations:
        for i in range(len(seq) - sequenceLength - 2):
            train.append(seq[i:i+sequenceLength])
            target.append(seq[i+sequenceLength+1])
    
    #Define number of samples, notes and chords, and durations
    nSamples = len(train)
    print("Total number of Samples is ", nSamples)
    train = np.asarray(train)
    target = np_utils.to_categorical(target, num_classes=num_combinations)
    
    return (train, target)


def create_model(args, num_tokens):

    sequenceLength = args.sequence_length
    embedDim = args.embed_dim

    # Define input layers
    lstm_input = Input(shape = (sequenceLength,))
    
    # Define embedding layers
    embed_layer = Embedding(num_tokens, embedDim, input_length = sequenceLength)(lstm_input)
    lstm_layer = LSTM(512)(embed_layer)
    bn_layer = BatchNorm()(lstm_layer)
    dropout_layer = Dropout(0.3)(bn_layer)
    output_layer = Dense(num_tokens, activation = 'softmax')(dropout_layer)

    # Define model
    lstm = Model(inputs = lstm_input, outputs = output_layer)
    # Compile the model
    lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop', loss_weights=[1.0, 0.4])
    
    print(lstm.summary())
    plot_model(lstm, "data/models/" + args.name + ".png", show_shapes=True)

    return lstm

def train_model(model, data, args, batch_size=64, type=None):
    print("Training lstm model with ", args.epochs, " epochs.")
    print("This is the ", args.model_type," model")

    #unpack the data
    train, target = data

    filepath = "tmp/checkpoints/combination/1/weights.{epoch:02d}.hdf5"
        
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]
    model_name = args.name

    if args.checkpoint:
        model.fit(train, target, epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1, initial_epoch=args.initial_epoch)
    else:
        model.fit(train, target, epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    path = "data/models/" + args.name
    model.save(path)
    print("Just saved model to ", path)

if __name__ == '__main__':

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    train_network()
