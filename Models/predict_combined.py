""" 
This module generates predicted sequences in the testing set using the 
LSTM model with the combined feature representation.  This feature representation is a tuple of notes and durations that is mapped to an integer value using the dictionary created by calling the data.py file.  The predictions are compared to the gold truth and a score is calculated for the accuracy of the predictions.
"""

import pickle
import glob
import os
import csv
import random
import utils
import copy
import numpy as np
from music21 import converter, instrument, note, stream, chord, pitch, interval, key
from keras.models import load_model
from keras.utils import np_utils

def generate():
    print("Getting test data")
    args = utils.get_args()

    #Load pickled dicts and test file names
    path = args.data_file + "testList"
    with open(path, 'rb') as filepath:
        testFileNames = pickle.load(filepath)
    
    path = args.data_file + "noteDurToInt"
    with open(path, 'rb') as filepath:
        noteDurToInt = pickle.load(filepath)
        print("The number of unique combinations used for this data is: ", len(noteDurToInt))
    
    data = get_test_data(testFileNames, noteDurToInt, args)
    targetSequences = predict_sequences(data, noteDurToInt, args)
    score = evaluate(targetSequences, testFileNames, args)
    print("The final score is ", score)

def get_test_data(testFileNames, noteDurToInt, args):
    
    sequenceLength = args.sequence_length
    primeScores = []
    primeKeys = []

    for fname in testFileNames:
        primeFilePath = os.path.join(args.data_file + "prime_midi/", fname)
        try:
            primeMidi = converter.parse(primeFilePath)
        except:
            print("File would not parse correctly: ", fname)
        
        if args.transpose == "C":
            primeKeys.append(primeMidi.analyze('key'))
            primeMidi = utils.transposeToC(primeMidi)
        
        primeScores.append(primeMidi)
    
    combinations = [[] for _ in primeScores]
  
    # Extract notes, chords, durations, and keys
    for i, song in enumerate(primeScores):
        if i % 10 == 0:
            print ("Processing song no ", i)
        for element in song.flat:
            if isinstance(element, note.Note):
                combinations[i].append((element.pitch.midi, element.duration.quarterLength))
            elif isinstance(element, note.Rest):
                combinations[i].append((128, element.duration.quarterLength))
        
        combinations[i] = combinations[i][-sequenceLength:]

    for j, seq in enumerate(combinations):
        for k, element in enumerate(seq):
            try:
                combinations[j][k] = noteDurToInt[element]
            except:
                print("Combination ", element, " not found in dictionary!")
                #Didn't find combination in dict, just use 0.5 and 128
                combinations[j][k] = noteDurToInt[(128, 0.5)]

    return (combinations, primeKeys) 

def predict_sequences(data, noteDurToInt, args):
    print("Predicting sequences...")

    modelPath = args.saved_model
    sequenceLength = args.sequence_length
    model = load_model(modelPath)
    
    #unpack data
    primes, keys = data

    num_combinations = len(noteDurToInt)
    #Invert dictionaries
    intToDurNote = {i: c for c, i in noteDurToInt.items()}
    predictedNotes = []
    predictedDurations = []
    predictedOffsets = []

    # generate 10 notes
    for i in range(len(primes)):
        note_seq = []
        dur_seq = []
        offset_seq = []
        input = np.expand_dims(primes[i],0)
    
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        offset = 0
        while numBeats <= durationLength:
        #for note_index in range(10):
            prediction = np.argmax(model.predict(input))
            note, dur = intToDurNote[prediction]
            if note != 128:
                note_seq.append(note)
                dur_seq.append(dur)
                offset_seq.append(offset)
            offset +=  dur
            input[0][:-1] = input[0][1:]
            input[0][-1] = prediction
            numBeats += dur

        predictedNotes.append(note_seq)
        predictedDurations.append(dur_seq)
        predictedOffsets.append(offset_seq)

    return predictedNotes, predictedOffsets, predictedDurations


def evaluate(targetSequences, fileNames, args):
    """
    Evaluates the accuracy of the prediction using a cardinality score and a pitch score.
    Each of the two scores is weighted equally, and the final score is the total.
    The total maximum score is 1.0.
    """
    print("Calculating scores")
    #unpack data
    predictedNotes, predictedOffsets, predictedDurations = targetSequences
    
    numSongs = len(predictedNotes)
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
        for j in range(len(predictedNotes[i])):
            predictions.append((predictedOffsets[i][j], predictedNotes[i][j]))
            
        #print some samples
        #if i % 10 != 0:
#            print("Calculating score for song no ", i)
#            print("True Notes:")
#            print(trueNotes)
#            print("Predicted Notes:")
#            print(predictedNotes[i])
#            print("Predicted Durations")
#            print(predictedDurations[i])
        
        # Calculate Score
        cScore = utils.cardinalityScore(true, predictions)
        pScore = utils.pitchScore(trueNotes, predictedNotes[i])

        totalScore = cScore * 0.5 + pScore * 0.5
        totalScores[i] = totalScore
        totalCScores[i] = cScore * 0.5
        totalPScores[i] = pScore * 0.5

    print("Total Cardinality Score is: ", np.sum(totalCScores) / numSongs)
    print("Total Pitch Score is: ", np.sum(totalPScores) / numSongs)

    return np.sum(totalScores) / numSongs

if __name__ == '__main__':
    generate()
