""" 
This module creates an LSTM model to be used with the  midi data files
in the project.  The LSTM can be either a multivariate model that takes
two inputes:  notes and durations, which are both represented as integers, or a parallel model that trains note and duration sequences on two separate LSTM models.. 
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
   
    if args.music_type == "mono":
        num_notes = 130
        notesToInt = None
    else:
        #get dictionary of chords
        with open(args.data_file + 'notesToInt', 'rb') as filepath:
            notesToInt = pickle.load(filepath)
        num_notes = len(notesToInt)
        print("The number of unique notes/chords is ", len(notesToInt))

    #get dictionary of durations
    with open(args.data_file + 'durationToInt', 'rb') as filepath:
        durationsToInt = pickle.load(filepath)
    #print(durationsToInt)
    num_durations = len(durationsToInt)
    
    #Get data from midi files
    data = process_data(args, notesToInt, durationsToInt)
    
    #Get model
    if args.checkpoint == None:

        if args.model_type == "parallel":
            model_notes = create_model_parallel(args, num_notes)
            train_model(model_notes, data, args, type="notes")
            model_durations = create_model_parallel(args, num_durations)
            train_model(model_durations, data, args, type="durations")
            
        #model_type == "single"
        else:
            model = create_model(args, num_notes, num_durations)
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

def process_data(args, notesToInt, durationsToInt):
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
    notes = [[] for _ in scores]
    durations = [[] for _ in scores]
  
    if args.music_type == "mono":
        # Extract notes, chords, durations, and keys
        print("Processing mono music files")
        for i, song in enumerate(scores):
            if i % 10 == 0:
                print ("Processing song no ", i)
            for element in song.flat:
                if isinstance(element, note.Note):
                    notes[i].append(element.pitch.midi)
                    durations[i].append(element.duration.quarterLength)
                elif isinstance(element, note.Rest):
                    notes[i].append(128)
                    durations[i].append(element.duration.quarterLength)

    #music type is "poly"
    else:
        print("Processing polyphonic music files")
        # Extract notes, chords, durations, and keys
        for i, song in enumerate(scores):
            if i % 10 == 0:
                print ("Processing song no ", i)
            newStream = stream.Stream()
            for part in song.parts:
                newStream.mergeElements(part)
            for element in newStream.chordify():
                if isinstance(element, note.Note):
                    notes[i].append(str(element.pitch.midi))
                    durations[i].append(element.duration.quarterLength)
                elif isinstance(element, note.Rest):
                    notes[i].append("128")
                    durations[i].append(element.duration.quarterLength)
                elif isinstance(element, chord.Chord):
                    chords = ".".join(str(n.midi) for n in element.pitches)
                    notes[i].append(chords)
                    durations[i].append(element.duration.quarterLength)
            
            notes[i] = [notesToInt[n] for n in notes[i]]

    sequenceLength = args.sequence_length

    # Define empty arrays for train data
    trainNotes = []
    trainDurations = []
    targetNotes = []
    targetDurations = []
    num_notes = 130
    if args.music_type == "poly":
        num_notes = len(notesToInt)
    num_durations = len(durationsToInt)
    
    # Construct training sequences for chords and durations
    for noteSeq, durSeq in zip(notes, durations):
        durSeq = [durationsToInt[d] for d in durSeq]
        for i in range(len(noteSeq) - sequenceLength - 2):
            trainNotes.append(noteSeq[i:i+sequenceLength])
            trainDurations.append(durSeq[i:i+sequenceLength])
            targetNotes.append(noteSeq[i+sequenceLength+1])
            targetDurations.append(durSeq[i+sequenceLength+1])    
    
    #Define number of samples, notes and chords, and durations
    nSamples = len(trainNotes)
    print("Total number of Samples is ", nSamples)
    trainNotes = np.asarray(trainNotes)
    trainDurations = np.asarray(trainDurations)
    targetNotes = np_utils.to_categorical(targetNotes, num_classes=num_notes)
    targetDurations = np_utils.to_categorical(targetDurations, num_classes=num_durations)
    
   
    return (trainNotes, targetNotes, trainDurations, targetDurations)


def create_model(args, num_notes, num_durations):

    sequenceLength = args.sequence_length
    embedDim = args.embed_dim

    # Define input layers
    noteInput = Input(shape = (sequenceLength,), name="notes")
    durationInput = Input(shape = (sequenceLength,), name="durations")
    
    # Define embedding layers
    noteEmbedding = Embedding(num_notes, embedDim, input_length = sequenceLength)(noteInput)
    durationEmbedding = Embedding(num_durations, embedDim, input_length = sequenceLength)(durationInput)

    #Reduce sequence of embedded chords into a single 512-dimensional vector
    noteLSTM = LSTM(512)(noteEmbedding)
    #Reduce sequence of embedded chords into a single 128-dimensional vector
    durationLSTM = LSTM(128)(durationEmbedding)

    noteBN = BatchNorm()(noteLSTM)
    durationBN = BatchNorm()(durationLSTM)

    noteFeatures = Dropout(0.3)(noteBN)
    durationFeatures = Dropout(0.3)(durationBN)
    
    
    # Merge all available features into a single large vector ia concatenation
    mergeLayer = Concatenate(axis=1)([noteFeatures, durationFeatures])
    
    # Define output layers
    noteOutput = Dense(num_notes, activation = 'softmax')(mergeLayer)
    durationOutput = Dense(num_durations, activation = 'softmax')(mergeLayer)

    # Define model
    lstm = Model(inputs = [noteInput, durationInput], outputs = [noteOutput, durationOutput])#
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
    trainNotes, targetNotes, trainDurations, targetDurations = data

    if type == "notes":
        filepath = "tmp/checkpoints/notes/weights.{epoch:02d}.hdf5"
    elif type == "durations":
        filepath = "tmp/checkpoints/durations/weights.{epoch:02d}.hdf5"
    else:
        filepath = "tmp/checkpoints/weights.{epoch:02d}.hdf5"
        
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
        model.fit([trainNotes, trainDurations], [targetNotes, targetDurations], epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1, initial_epoch=args.initial_epoch)
    
    if args.checkpoint and args.model_type == "parallel":
        if (args.checkpoint_model == "notes" and type == "notes") or args.checkpoint_model == "durations":
            model.fit(input_layer, output_layer, epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1, initial_epoch=args.initial_epoch)
        #This is the case where the notes checkpoint was used but the durations didn't have any checkpoints yet.
        else:

            model.fit(input_layer, output_layer, epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    if not args.checkpoint and args.model_type == "parallel":
        model.fit(input_layer, output_layer, epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    if not args.checkpoint and args.model_type == "single": 
        model.fit([trainNotes, trainDurations], [targetNotes, targetDurations], epochs=args.epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    path = "data/models/" + model_name
    model.save(path)
    print("Just saved model to ", path)
    #with open(path + '-history', 'wb') as filepath:
    #    pickle.dump(history, filepath)

if __name__ == '__main__':

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    train_network()
