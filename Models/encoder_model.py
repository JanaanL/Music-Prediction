""" 
This module creates an LSTM Encoder-Decoder model to be used with the  midi data files
in the project.  The Encoder-Decoder model is a multivariate model that takes
two inputes:  notes and durations, which are both represented as integers. 
The largest input sequence in the training data is the input dimension of the encoder model.
The largest target sequence of the training data is one of the input dimensions of the decoder model.
"""

import glob
import pickle
import numpy as np
import os
import math
import argparse
import random
import utils
from music21 import converter, instrument, note, chord, stream, interval, pitch
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import TimeDistributed
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
    
    with open(args.data_file + 'durationToInt', 'rb') as filepath:
        durationToInt = pickle.load(filepath)
    num_durations = len(durationToInt)

    data, seq_lengths = process_data(args, durationToInt)

    #Get model
    if args.checkpoint == None:

        model = create_model(args, num_durations, seq_lengths)
        train_model(model, data, args)

    #We are checkpointing
    else:
        #Load Checkpointed model
        print("Resume training at epoch :", args.initial_epoch)
        model = load_model(args.checkpoint)
        train_model(model, data, args)

def process_data(args, durationToInt):
    """ Get all the notes and chords from the midi files in the data_file directory """
    # Create empty list for scores
    path = args.data_file + "trainList"
    with open(path, 'rb') as filepath:
        trainFileNames = pickle.load(filepath)

    primeScores = []
    targetScores = []
    for fname in trainFileNames:
        primeFilePath = os.path.join(args.data_file + "prime_midi/",fname)
        targetFilePath = os.path.join(args.data_file + "cont_true_midi/",fname)
        try:
            primeMidi = converter.parse(primeFilePath)
            targetMidi = converter.parse(targetFilePath)
            primeMidi.flat
            targetMidi.flat
            
            if args.transpose == "all":
                transposedPrime = utils.transposeToAll(primeMidi)
                primeScores = primeScores + transposedPrime
                transposedTarget = utils.transposeToAll(targetMidi)
                targetScores = targetScores + transposedTarget
            
            if args.transpose == "C":
                primeMidi = utils.transposeToC(primeMidi)
                targetMidi = utils.transposeToC(targetMidi)

            primeScores.append(primeMidi)
            targetScores.append(targetMidi)
        
        except:
            print("File would not parse correctly: ", fname)

    def process_notes(scores, target=False):
        notes = [[] for _ in scores]
        durations = [[] for _ in scores]
        keys = []
        
        # Extract notes, chords, durations, and keys
        for i, song in enumerate(scores):
            if target:
                #append starting sequence value of 1 to beginning of target sequence
                notes[i].append(1)
                durations[i].append("B")
            keys.append(str(song.analyze('key')))
            for element in song.flat:
                if isinstance(element, note.Note):
                    notes[i].append(element.pitch.midi)
                    durations[i].append(element.duration.quarterLength)
                elif isinstance(element, note.Rest):
                    notes[i].append(128)
                    durations[i].append(element.duration.quarterLength)
            #append ending sequence value of 129 to end of target sequence
            if target:
                notes[i].append(129)
                durations[i].append("E")

            durations[i] = [durationToInt[d] for d in durations[i]]
        return notes, durations, keys

    primeNotes, primeDurations, keys = process_notes(primeScores)
    targetNotes, targetDurations, keys = process_notes(targetScores, target=True)
    
    num_durations = len(durationToInt)
    print("Length of unique durations: ", num_durations)

    max_encoder_seq_length = max([len(notes) for notes in primeNotes])
    max_decoder_seq_length = max([len(txt) for txt in targetNotes])

    print("Number of samples:", len(primeNotes))
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)

    def process_sequences(primes, targets, type="notes"):

        num_tokens = 130
        if type == "durations":
            num_tokens = num_durations
    
        max_prime_seq = max([len(seq) for seq in primes])
        max_target_seq = max([len(seq) for seq in targets])
        target_output = np.zeros((len(targets), max_target_seq, num_tokens))

        for i, (prime, target) in enumerate(zip(primes, targets)):
            for t in range(len(prime), max_prime_seq):
                prime.append(0)
            for t in range(len(target), max_target_seq):
                target.append(0)
            #one-hot encode the decoder output
            for t in range(0, max_target_seq-1):
                elem = targets[i][t+1]
                target_output[i][t][elem] = 1.0
        
        primes = np.asarray(primes)
        targets = np.asarray(targets)
        return primes, targets, target_output

    encoder_input_data_1, decoder_input_data_1, decoder_target_data_1 = process_sequences(primeNotes, targetNotes)
    encoder_input_data_2, decoder_input_data_2, decoder_target_data_2 = process_sequences(primeDurations, targetDurations, type="durations")

    return (encoder_input_data_1, encoder_input_data_2, decoder_input_data_1, decoder_input_data_2, decoder_target_data_1, decoder_target_data_2), (max_encoder_seq_length, max_decoder_seq_length)

def create_model(args, num_durations, seq_lengths):

    encoder_seq_length, decoder_seq_length = seq_lengths
    num_notes = 130
    embedDim = args.embed_dim

    # Define encoder for training
    encoder_inputs_1 = Input(shape = (encoder_seq_length,), name="notes")
    encoder_inputs_2 = Input(shape = (encoder_seq_length,), name="durations")
    # Define embedding layers
    noteEmbedding = Embedding(num_notes, embedDim, input_length = encoder_seq_length)(encoder_inputs_1)
    durationEmbedding = Embedding(num_durations, embedDim, input_length = encoder_seq_length)(encoder_inputs_2)
    merge_layer = Concatenate(axis=1)([noteEmbedding, durationEmbedding])
    merge_encoder = LSTM(512, return_state=True)
    encoder_outputs, state_h, state_c = merge_encoder(merge_layer)
    encoder_states = [state_h, state_c]

    # Define decoder for training
    decoder_inputs_1 = Input(shape=(decoder_seq_length,))
    decoder_embedding = Embedding(num_notes, embedDim, input_length = decoder_seq_length)(decoder_inputs_1)
    decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_outputs_1 = Dense(num_notes, activation="softmax")(decoder_outputs)
    decoder_outputs_2 = Dense(num_durations, activation="softmax")(decoder_outputs)

    model = Model([encoder_inputs_1, encoder_inputs_2, decoder_inputs_1], [decoder_outputs_1, decoder_outputs_2])

    print(model.summary())
    plot_model(model, "data/models/" + args.name + ".png", show_shapes=True)
    return model

    print("Training encoder-decoder model with ", args.epochs, " epochs.")
    
    #unpack the data
    encoder_input_data_1, encoder_input_data_2, decoder_input_data_1, decoder_input_data_2, decoder_target_data_1, decoder_target_data_2 = data

    # Train the model
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    filepath = "tmp/checkpoints/s2s/4/weights.{epoch:02d}.hdf5"
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
        history = model.fit([encoder_input_data_1, encoder_input_data_2, decoder_input_data_1, decoder_input_data_2], [decoder_target_data_1, decoder_target_data_2], batch_size=batch_size, epochs=args.epochs, callbacks=callbacks_list, initial_epoch=args.initial_epoch, validation_split=0.1)

    else:
        history = model.fit([encoder_input_data_1, encoder_input_data_2, decoder_input_data_1, decoder_input_data_2], [decoder_target_data_1, decoder_target_data_2], batch_size=64, epochs=args.epochs, callbacks=callbacks_list, validation_split=0.1)

    #print(history.history.keys())
    path = "data/models/" + model_name
    model.save(path)
    #with open(path + 'history', 'wb') as filepath:
    #        pickle.dump(history)

if __name__ == '__main__':

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    train_network()
