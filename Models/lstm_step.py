""" 
This module creates an LSTM model that uses an interval representation for the noteinput.  The note input is represented as a sequence of intervals.  These are the intervals between the note pitches in the note input.  The duration input is a sequence of durations that are mapped to integer values using  a dictionary that was created by calling the data.py file.  
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
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
import tensorflow as tf


def train_network():
    args = utils.get_args()
   
    num_steps = 260
    #get dictionary of durations
    with open(args.data_file + 'durationToInt', 'rb') as filepath:
        durationsToInt = pickle.load(filepath)
    num_durations = len(durationsToInt)
    print("Durations to Int")
    print(durationsToInt)

    #Get data from midi files
    data = process_data(args, durationsToInt)
    
    #Get model
    if args.checkpoint == None:

        if args.model_type == "parallel":
            model_notes = create_model_parallel(args, num_notes)
            train_model(model_notes, data, args, type="notes")
            model_durations = create_model_parallel(args, num_durations)
            train_model(model_durations, data, args, type="durations")
            
        #model_type == "single"
        else:
            model = create_model(args, num_steps, num_durations)
            train_model(model, data, args)
    
    #We are checkpointing
    else:
        print("Resume training at epoch :", args.initial_epoch)
        if args.model_type == "parallel":
            if args.checkpoint_model == "notes":
                model_notes = load_model(args.checkpoint)
                train_model(model_notes, data, args, type="notes")
                model_durations = create_model_parallel(args, num_durations)
                train_model(model_durations, data, args, type="durations")
            
            #Only the durations model is being checkpointed
            else:
                model_durations = load_model(args.checkpoint)
                train_model(model_durations, data, args, type="durations")
        
        #The single model is being checkpointed
        else:
            model = load_model(args.checkpoint)
            train_model(model, data, args)

def process_data(args, durationsToInt):
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
        
    print(trainFileNames[0])
    steps = [[] for _ in scores]
    durations = [[] for _ in scores]
    keys = []
  
    firstElement = True 
    # Extract notes, chords, durations, and keys
    for i, song in enumerate(scores):
        if i % 10 == 0:
            print ("Processing song no ", i)
        for element in song.flat:
            if isinstance(element, note.Note):
                if not firstElement:
                    diff = element.pitch.midi - priorNote + 130
                    steps[i].append(diff)
                    durations[i].append(element.duration.quarterLength)
                else:
                    firstElement = False
                priorNote = element.pitch.midi
                
            elif isinstance(element, note.Rest):
                if not firstElement:
                    steps[i].append(130)
                    durations[i].append(element.duration.quarterLength)

    sequenceLength = args.sequence_length

    # Define empty arrays for train data
    trainSteps = []
    trainDurations = []
    targetSteps = []
    targetDurations = []
    num_steps = 260
    num_durations = len(durationsToInt)
    
    # Construct training sequences for chords and durations
    for stepSeq, durSeq in zip(steps, durations):
        durSeq = [durationsToInt[d] for d in durSeq]
        for i in range(len(stepSeq) - sequenceLength - 2):
            trainSteps.append(stepSeq[i:i+sequenceLength])
            trainDurations.append(durSeq[i:i+sequenceLength])
            targetSteps.append(stepSeq[i+sequenceLength+1])
            targetDurations.append(durSeq[i+sequenceLength+1])    
    
    #Define number of samples, notes and chords, and durations
    nSamples = len(trainSteps)
    print("Total number of Samples is ", nSamples)
    trainSteps = np.asarray(trainSteps)
    trainDurations = np.asarray(trainDurations)
    targetSteps = np_utils.to_categorical(targetSteps, num_classes=num_steps)
    targetDurations = np_utils.to_categorical(targetDurations, num_classes=num_durations)
    
   
    return (trainSteps, targetSteps, trainDurations, targetDurations)


def create_model(args, num_steps, num_durations):

    sequenceLength = args.sequence_length
    embedDim = args.embed_dim

    # Define input layers
    stepInput = Input(shape = (sequenceLength,), name="steps")
    durationInput = Input(shape = (sequenceLength,), name="durations")
    
    # Define embedding layers
    stepEmbedding = Embedding(num_steps, embedDim, input_length = sequenceLength)(stepInput)
    durationEmbedding = Embedding(num_durations, embedDim, input_length = sequenceLength)(durationInput)

    #Reduce sequence of embedded chords into a single 512-dimensional vector
    stepLSTM = LSTM(512)(stepEmbedding)
    #Reduce sequence of embedded chords into a single 128-dimensional vector
    durationLSTM = LSTM(128)(durationEmbedding)

    stepBN = BatchNorm()(stepLSTM)
    durationBN = BatchNorm()(durationLSTM)

    stepFeatures = Dropout(0.3)(stepBN)
    durationFeatures = Dropout(0.3)(durationBN)
    
    
    # Merge all available features into a single large vector ia concatenation
    mergeLayer = Concatenate(axis=1)([stepFeatures, durationFeatures])
    
    # Define output layers
    stepOutput = Dense(num_steps, activation = 'softmax')(mergeLayer)
    durationOutput = Dense(num_durations, activation = 'softmax')(mergeLayer)

    # Define model
    lstm = Model(inputs = [stepInput, durationInput], outputs = [stepOutput, durationOutput])#
    # Compile the model
    lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop', loss_weights=[1.0, 0.4])
    
    print(lstm.summary())
    plot_model(lstm, "data/models/" + args.name + ".png", show_shapes=True)

    return lstm

def create_model_parallel(args, num_tokens):

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
    if args.model_type == "parallel":
        print("This is the ", type, " lstm")

    #unpack the data
    trainSteps, targetSteps, trainDurations, targetDurations = data

    if type == "notes":
        filepath = "tmp/checkpoints/7/notes/weights.{epoch:02d}.hdf5"
    elif type == "durations":
        filepath = "tmp/checkpoints/lstm/7/durations/weights.{epoch:02d}.hdf5"
    else:
        filepath = "tmp/checkpoints/lstm/8/weights.{epoch:02d}.hdf5"
        
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]
    model_name = args.name

    if args.model_type == "parallel":
        if type == "notes":
            input_layer = trainNotes
            output_layer = targetNotes
            model_name += "-notes"
        elif type == "durations":
            input_layer = trainDurations
            output_layer = targetDurations
            model_name += "-durations"

    if args.checkpoint and args.model_type == "single":
        model.fit([trainSteps, trainDurations], [targetSteps, targetDurations], epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1, initial_epoch=args.initial_epoch)
    
    if args.checkpoint and args.model_type == "parallel":
        if (args.checkpoint_model == "notes" and type == "notes") or args.checkpoint_model == "durations":
            model.fit(input_layer, output_layer, epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1, initial_epoch=args.initial_epoch)
        #This is the case where the notes checkpoint was used but the durations didn't have any checkpoints yet.
        else:

            model.fit(input_layer, output_layer, epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    if not args.checkpoint and args.model_type == "parallel":
        model.fit(input_layer, output_layer, epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    if not args.checkpoint and args.model_type == "single": 
        model.fit([trainSteps, trainDurations], [targetSteps, targetDurations], epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    path = "data/models/" + model_name
    model.save(path)
    print("Just saved model to ", path)
    #with open(path + '-history', 'wb') as filepath:
    #    pickle.dump(history, filepath)

if __name__ == '__main__':

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    train_network()
