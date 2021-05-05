""" 
This module generates predicted notes and sequences in the testing set using the 
LSTM Encoder-Decoder model.  The predictions are compared to the gold truth and a 
score is calculated for the accuracy of the predictions.
"""

import pickle
import glob
import os
import csv
import argparse
import random
import copy
import utils
import numpy as np
from music21 import converter, instrument, note, stream, chord, pitch, interval, key
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.models import load_model
from keras.utils import np_utils

def generate():
    print("Getting test data")
    args = utils.get_args()

    #Load pickled dicts and test file names
    path = args.data_file + "testList"
    with open(path, 'rb') as filepath:
        testFileNames = pickle.load(filepath)

    path = args.data_file + "durationToInt"
    with open(path, 'rb') as filepath:
        durationToInt = pickle.load(filepath)
        print("The number of unique durations used for this data is: ", len(durationToInt))
   
    data = get_test_data(testFileNames, durationToInt, args)
    targetSequences = predict_sequences(data, durationToInt, args)
    notes, offsets, durations = create_midis(targetSequences, args)
    score = evaluate(notes, offsets, durations, testFileNames, args)

    print("The final score is ", score)

def get_test_data(testFileNames, durationToInt, args):

    primeScores = []
    for fname in testFileNames:
        primeFilePath = os.path.join(args.data_file + "prime_midi/",fname)
        try:
            primeMidi = converter.parse(primeFilePath)
        except:
            print("File would not parse correctly: ", fname)
        
        primeMidi.flat
        
        if args.transpose == "all":
            transposedPrime = utils.transposeToAll(primeMidi)
            primeScores = primeScores + transposedPrime
        
        if args.transpose == "C":
            primeMidi = utils.transposeToC(primeMidi)

        primeScores.append(primeMidi)

        notes = [[] for _ in primeScores]
        durations = [[] for _ in primeScores]
        keys = []
        
    # Extract notes, chords, durations, and keys
    for i, song in enumerate(primeScores):
        if i % 10 == 0:
            print("Processing file no. ", i)

        keys.append(str(song.analyze('key')))
        for element in song.flat:
            if isinstance(element, note.Note):
                notes[i].append(element.pitch.midi)
                durations[i].append(element.duration.quarterLength)
            elif isinstance(element, note.Rest):
                notes[i].append(128)
                durations[i].append(element.duration.quarterLength)

        for j, dur in enumerate(durations[i]):
            try:
                durations[i][j] = durationToInt[dur]
            except:
                print("Duration ", dur, " not found in dictionary!")
                #Didn't find duration in dict, just use 0.5
                durations[i][j] = durationToInt[0.5]
    
    max_seq_length = max([len(seq) for seq in notes])
    print("The maximum input length is ", max_seq_length)
    
    def process_sequences(primes, type="notes"):
        num_tokens = 130
        if type == "durations":
            num_tokens = len(durationToInt)

        max_seq_length = max([len(seq) for seq in primes])

        for i, seq in enumerate(primes):
            for t in range(len(seq), max_seq_length):
                seq.append(0)
        encoder_input_data = np.asarray(primes)
        return encoder_input_data

    encoder_input_data_1 = process_sequences(notes)
    encoder_input_data_2 = process_sequences(durations, type="durations")
    print("The shape of encoder_input_data_1 is ", encoder_input_data_1.shape)
    return (encoder_input_data_1, encoder_input_data_2, keys)
    

def predict_sequences(data, durationToInt, args):
    print("Predicting sequences...")

    modelPath = args.saved_model
    max_decoder_seq_length = 59
    num_notes = 130
    num_durations = len(durationToInt)
    intToDuration = {i: c for c, i in durationToInt.items()}

    #unpack data
    encoder_input_data_1, encoder_input_data_2, keys = data
    
    sample_size, seq_length = encoder_input_data_1.shape
    
    model = load_model(modelPath)
    encoder_inputs_1 = model.input[0]  # input_1
    encoder_inputs_2 = model.input[1]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[7].output  # lstm
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model([encoder_inputs_1, encoder_inputs_2], encoder_states)
     
    decoder_inputs = model.input[2]  # input_3
    decoder_state_input_h = Input(shape=(512,), name="input_4")
    decoder_state_input_c = Input(shape=(512,), name="input_5")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embedding = model.layers[6](decoder_inputs)
    decoder_lstm = model.layers[8]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs_1 = model.layers[9](decoder_outputs)
    decoder_outputs_2 = model.layers[10](decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs_1, decoder_outputs_2] + decoder_states)
     
 
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)
                 
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start number.
        target_seq[0, 0] = 1.0
         
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_notes = []
        decoded_durations = []
        while not stop_condition:
            output_tokens_1, output_tokens_2,  h, c = decoder_model.predict([target_seq] + states_value)
                 
            # Sample a token
            sampled_note = np.argmax(output_tokens_1[0, -1, :])
            print("sampled note:", sampled_note)
            sampled_dur = np.argmax(output_tokens_2[0, -1, :])
            decoded_notes.append(sampled_note)
            decoded_durations.append(sampled_dur)
                                     
            # Exit condition: either hit max length
            # or find stop character.
            if sampled_note == 129 or len(decoded_notes) > max_decoder_seq_length:
                stop_condition = True
                                         
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_note
                 
            # Update states
            states_value = [h, c]
        return (decoded_notes, decoded_durations)

    #Iterate through input sequences to generate output sequences
    targetNotes = []
    targetDurations = []

    for i in range(sample_size):
        if i % 10 == 0:
            print("Predicting sequence for song no ", i)
        input_seq_1 = np.reshape(encoder_input_data_1[i], (-1, seq_length))
        input_seq_2 = np.reshape(encoder_input_data_2[i], (-1, seq_length))
        predNotes, predDurations = decode_sequence([input_seq_1, input_seq_2])
        targetNotes.append(predNotes)
        targetDurations.append(predDurations)

    #Converte duration sequences to duration values
    for j in range(len(targetDurations)):
        targetDurations[j] = [intToDuration[d] for d in targetDurations[j]]
    
    return (targetNotes, targetDurations)

def create_midis(targetSequences, args):
    """ Converts the output from the prediction to a midi file"""

    targetNotes, targetDurations = targetSequences
    midiFiles = []
    count = 0
    for i in range(len(targetNotes)):
        
        if i % 10 == 0:
            print("Creating midi file for song no ", i)

        generatedStream = stream.Stream()
        generatedStream.append(instrument.Piano())

        # Add notes and durations to stream
        for j in range(len(targetNotes[i])):
            targetDuration = targetDurations[i][j]
            if targetDuration == "E"  or targetDuration == "B":
                targetDuration = 0.5
                count += 1
            if targetNotes[i][j] not in [0,1,128, 129]:
                newNote = note.Note(targetNotes[i][j])
                newNote.duration.quarterLength = targetDuration
                generatedStream.append(newNote)
            elif targetNotes[i][j] == 128:
                rest = note.Rest()
                rest.duration.quarterLength = targetDuration
                generatedStream.append(rest)

        if args.transpose == "C":
            mode = primeKeys[i].mode
            if mode == 'major':
                i = interval.Interval(pitch.Pitch('C'), primeKeys[i].tonic)
            elif mode == 'minor':
                i = interval.Interval(pitch.Pitch('c'), primeKeys[i].tonic)

            generatedStream = generatedStream.flat.transpose(i)

        #generatedStream.show('text')
        midiFiles.append(generatedStream)
    
    notes = [[] for _ in midiFiles]
    durations = [[] for _ in midiFiles]
    offsets = [[] for _ in midiFiles]

    for i, midi in enumerate(midiFiles):
        for element in midi.flat:
            if isinstance(element, note.Note):
                notes[i].append(element.pitch.midi)
                durations[i].append(element.duration.quarterLength)
                offsets[i].append(element.offset)

    return notes, offsets, durations


def evaluate(notePredictions, offsetPredictions, durationPredictions, fileNames, args):
    """
    Evaluates the accuracy of the prediction using a cardinality score and a pitch score.
    Each of the two scores is weighted equally, and the final score is the total.
    The total maximum score is 1.0.
    """

    print("Calculating scores")
    numSongs = len(notePredictions)
    totalScores = np.zeros(numSongs)
    totalCScores = np.zeros(numSongs)
    totalPScores = np.zeros(numSongs)

    for i in range(numSongs):
        trueNotes = []
        trueOffsets = []
        true = []
        predictions = []
        file = fileNames[i].replace(".mid", "")
        path = os.path.join(args.data_file + "cont_true_csv/", file + ".csv")
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                trueOffsets.append(float(row[0]))
                trueNotes.append(int(row[1]))
                true.append((float(row[0]), int(row[1])))
        for j in range(len(notePredictions[i])):
            predictions.append((offsetPredictions[i][j], notePredictions[i][j]))
            
        
        #print some samples
        #if i % 10 != 0:
#            print("Calculating score for song no ", i)
#            print("True Notes:")
#            print(trueNotes)
#            print("Predicted Notes:")
#            print(notePredictions[i])
#            print("Predicted Durations")
#            print(durationPredictions[i])
        
        
        # Calculate Score
        cScore = utils.cardinalityScore(true, predictions)
        pScore = utils.pitchScore(trueNotes, notePredictions[i])

        totalScore = cScore * 0.5 + pScore * 0.5
        totalScores[i] = totalScore
        totalCScores[i] = cScore * 0.5
        totalPScores[i] = pScore * 0.5

    print("Total Cardinality Score is: ", np.sum(totalCScores) / numSongs)
    print("Total Pitch Score is: ", np.sum(totalPScores) / numSongs)

    return np.sum(totalScores) / numSongs


if __name__ == '__main__':
    generate()
