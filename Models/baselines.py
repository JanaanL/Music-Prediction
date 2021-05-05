"""
This module calculates baseline scores for the different datasets
Three different types of baselines are calculated:  random, sampled, and sampled with constraints.

Random baseline sequences are sequences of notes/chords and durations chosen randomly from the training set.

Sampled sequences are sequences of notes/chords that are sampled from the training set using the distribution
of notes/chords from the training set.

Samples with constraint sequences are sequences of notes/chords that are sampled from the training set using
the distribution of notes/chords from the training set.  However, the sampled note must be in the prime sequence
of the song that is predicted.  If the predicted note is not in the prime, another note is sampled until one is 
found that is in the sequence.

Both types of datasets (monophonic and polyphonic) can be used with this module and must be specified as a 
command-line argument.

"""

import pickle
import glob
import os
import csv
import random
import argparse
import utils
import numpy as np
from music21 import converter, instrument, note, stream, chord, pitch, interval, key

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="../Data/mono_small/", help="path of data files to load")
    parser.add_argument("--music-type", type=str, default = "mono", choices=["poly", "mono"], help="Type of music that will be used.  Choices includes 'mono' for monophonic music and 'poly' for polyphonic music")
    args = parser.parse_args()
    
    print("Generating baselines for ", args.music_type, " datasets")

    #Load pickled dicts and train and test file names
    path = args.data_file + "trainList"
    with open(path, 'rb') as filepath:
        trainFileNames = pickle.load(filepath)
    path = args.data_file + "testList"
    with open(path, 'rb') as filepath:
        testFileNames = pickle.load(filepath)
    
    notesToInt = None
    if args.music_type == "poly":
        path = args.data_file + "notesToInt"
        with open(path, 'rb') as filepath:
            notesToInt = pickle.load(filepath)
            print("The number of unique notes/chords used for this data is: ", len(notesToInt))

    path = args.data_file + "durationToInt"
    with open(path, 'rb') as filepath:
        durationToInt = pickle.load(filepath)
        print("The number of unique durations used for this data is: ", len(durationToInt))
    
    data = get_training_data(trainFileNames, notesToInt, durationToInt, args)

    for predict_type in ["random", "sampled", "constrained"]:
        if predict_type == "random":
            predictions = random_prediction(notesToInt, durationToInt, len(testFileNames), args)
        elif predict_type == "sampled":
            predictions = sampled_prediction(data, notesToInt, durationToInt, len(testFileNames), args) 
        else:
            predictions = sampled_prediction_with_constraints(data, notesToInt, durationToInt, testFileNames, args)

        if args.music_type == "poly":
            predictions = process_poly(predictions, args) 
    
        score = evaluate(predictions, testFileNames, args)
        print("The total score is for the ", predict_type, " is ", score)


def get_training_data(trainFileNames, notesToInt, durationToInt, args):
    
    primeScores = []

    for fname in trainFileNames:
        filePath = os.path.join(args.data_file + "prime_midi/", fname)
        try:
            midi_in = converter.parse(filePath)
            midi_out = converter.parse(os.path.join(args.data_file + "cont_true_midi/", fname))
            midi_in.append(midi_out)
            midi_in.flat

            primeScores.append(midi_in)
        
        except:
            print("File would not parse correctly: ", fname)

    notes = [[] for _ in primeScores]
    durations = [[] for _ in primeScores]

    if args.music_type == "mono":

        # Extract notes, durations, offsets and keys
        for i, song in enumerate(primeScores):
            if i % 10 == 0:
                print("Processing file no. ", i)
            
            for element in song.flat:
                if isinstance(element, note.Note):
                    notes[i].append(element.pitch.midi)
                    durations[i].append(element.duration.quarterLength)
                elif isinstance(element, note.Rest):
                    notes[i].append(128)
                    durations[i].append(element.duration.quarterLength)
            
            for j, d in enumerate(durations[i]):
                durations[i][j] = durationToInt[d]
 
    # music type is "poly"
    else:
        # Extract notes, durations, offsets and keys
        for i, song in enumerate(primeScores):
            if i % 10 == 0:
                print("Processing file no. ", i) 
            newStream = stream.Stream()
            for part in song.parts:
                newStream.mergeElements(part)
            for element in newStream.chordify():
                if isinstance(element, note.Note):
                    notes[i].append(str(element.pitch.midi))
                    durations[i].append(element.duration.quarterLength)
                elif isinstance(element, note.Rest):
                    notes[i].append(str(128))
                    durations[i].append(element.duration.quarterLength)
                elif isinstance(element, chord.Chord):
                    c = ".".join(str(n.midi) for n in element.pitches)
                    notes[i].append(c)
                    durations[i].append(element.duration.quarterLength)
            
            for j, n in enumerate(notes[i]):
                notes[i][j] = notesToInt[n]
    
            for j, d in enumerate(durations[i]):
                durations[i][j] = durationToInt[d]
    
    return (notes, durations) 

def random_prediction(notesToInt, durationToInt, test_length, args):
    """
    This function generates a random sequence of notes/chords and durations.
    The predicted notes and durations are randomly chosen from the notes and 
    durations in the training set.
    """

    predictedNotes = []
    predictedDurations = []
    predictedOffsets = []

    #Invert dictionaries
    num_notes = 129
    if notesToInt:
        intToNotes = {i: c for c, i in notesToInt.items()}
        num_notes = len(notesToInt)
    intToDuration = {i: c for c, i in durationToInt.items()}
    nDurations = len(durationToInt)

    # generate 10 notes
    for i in range(test_length):
        contNotes = []
        contDurations = []
        contOffsets = []
    
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        offset = 0
        while numBeats <= durationLength:
            predNote = random.randint(0, num_notes-1)
            predDuration = random.randint(0, nDurations-1)
            contNotes.append(predNote)
            contDurations.append(predDuration)
            contOffsets.append(offset)
            numBeats += intToDuration[predDuration]
            offset += intToDuration[predDuration]

        contDurations = [intToDuration[c] for c in contDurations]
        if args.music_type == "poly":
            contNotes = [intToNotes[c] for c in contNotes]
        predictedNotes.append(contNotes)
        predictedDurations.append(contDurations)
        predictedOffsets.append(contOffsets)

    return (predictedNotes, predictedDurations, predictedOffsets)

def sampled_prediction(data, notesToInt, durationToInt, test_length, args):
    """
    This function generates a sampled sequence of notes/chords and durations.
    The predicted notes and durations are sampled from the distribution of notes/chords
    and durations found in the training set.
    """

    #unpack data
    trainNotes, trainDurations = data

    if args.music_type == "mono":
        num_notes = 129
    else:
        num_notes = len(notesToInt)
    num_durations = len(durationToInt)
    
    #Get distributions
    noteDist = np.zeros(num_notes)
    durDist = np.zeros(num_durations)
    for i in range(len(trainNotes)):
        for n,d in zip(trainNotes[i], trainDurations[i]):
            noteDist[n] += 1
            durDist[d] += 1
    noteDist = noteDist / noteDist.sum(axis=0, keepdims=True)
    durDist = durDist / durDist.sum(axis=0, keepdims=True)

    predictedNotes = []
    predictedDurations = []
    predictedOffsets = []

    #Invert dictionaries
    if notesToInt:
        intToNotes = {i: c for c, i in notesToInt.items()}
    intToDuration = {i: c for c, i in durationToInt.items()}

    # generate 10 notes
    for i in range(test_length):
        contNotes = []
        contDurations = []
        contOffsets = []
    
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        offset = 0
        while numBeats <= durationLength:
            predNote = np.random.choice(num_notes, p=noteDist)
            predDuration = np.random.choice(num_durations, p=durDist)
            contNotes.append(predNote)
            contDurations.append(predDuration)
            contOffsets.append(offset)
            numBeats += intToDuration[predDuration]
            offset += intToDuration[predDuration]

        contDurations = [intToDuration[c] for c in contDurations]
        if args.music_type == "poly":
            contNotes = [intToNotes[c] for c in contNotes]
        predictedNotes.append(contNotes)
        predictedDurations.append(contDurations)
        predictedOffsets.append(contOffsets)

    return (predictedNotes, predictedDurations, predictedOffsets)

def sampled_prediction_with_constraints(data, notesToInt, durationToInt, testFileNames, args):
    """
    This function generates a sampled sequence of notes/chords and durations with constraints.
    The predicted notes and durations are sampled from the distribution of notes/chords
    and durations found in the training set.  However, the predicted note must be in the prime
    sequence of the song being predicted.  If it isn't, another prediction is sampled until
    one is found that is in the prime sequence.
    """

    #unpack data
    trainNotes, trainDurations = data

    #Get constraints
    primeScores = []
    for fname in testFileNames:
        primeFilePath = os.path.join(args.data_file + "prime_midi/", fname)
        try:
            primeMidi = converter.parse(primeFilePath)
        except:
            print("File would not parse correctly: ", fname)
        
        primeScores.append(primeMidi)

    noteSets = [set() for _ in primeScores]
    
    if args.music_type == "mono":
        for i, song in enumerate(primeScores):
            for element in song.flat:
                if isinstance(element, note.Note):
                    noteSets[i].add(element.pitch.midi)
                if isinstance(element, note.Rest):
                    noteSets[i].add(128)

    else:
        for i, song in enumerate(primeScores):
            newStream = stream.Stream()
            for part in song.parts:
                newStream.mergeElements(part)
            for element in newStream.chordify():
                if isinstance(element, note.Note):
                    n = str(element.pitch.midi)
                    try:
                        n = notesToInt[n]
                    except:
                        n = 0
                    noteSets[i].add(n)
                elif isinstance(element, note.Rest):
                    noteSets[i].add(notesToInt["128"])
                elif isinstance(element, chord.Chord):
                    c = ".".join(str(n.midi) for n in element.pitches)
                    try:
                        n = notesToInt[c]
                    except:
                        print("Note ", c, " not found in dictionary!")
                        n = 0
                    noteSets[i].add(n)
            
    if args.music_type == "mono":
        num_notes = 129
    else:
        num_notes = len(notesToInt)
    num_durations = len(durationToInt)
    
    #Get distributions
    noteDist = np.zeros(num_notes)
    durDist = np.zeros(num_durations)
    for i in range(len(trainNotes)):
        for n,d in zip(trainNotes[i], trainDurations[i]):
            noteDist[n] += 1
            durDist[d] += 1
    noteDist = noteDist / noteDist.sum(axis=0, keepdims=True)
    durDist = durDist / durDist.sum(axis=0, keepdims=True)

    predictedNotes = []
    predictedDurations = []
    predictedOffsets = []

    #Invert dictionaries
    if notesToInt:
        intToNotes = {i: c for c, i in notesToInt.items()}
    intToDuration = {i: c for c, i in durationToInt.items()}

    # generate 10 notes
    for i in range(len(testFileNames)):
        contNotes = []
        contDurations = []
        contOffsets = []
    
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        offset = 0
        while numBeats <= durationLength:
            predNote = np.random.choice(num_notes, p=noteDist)
            while predNote not in noteSets[i]:
                predNote = np.random.choice(num_notes, p=noteDist)
            predDuration = np.random.choice(num_durations, p=durDist)
            contNotes.append(predNote)
            contDurations.append(predDuration)
            contOffsets.append(offset)
            numBeats += intToDuration[predDuration]
            offset += intToDuration[predDuration]

        contDurations = [intToDuration[c] for c in contDurations]
        if args.music_type == "poly":
            contNotes = [intToNotes[c] for c in contNotes]
        predictedNotes.append(contNotes)
        predictedDurations.append(contDurations)
        predictedOffsets.append(contOffsets)

    return (predictedNotes, predictedDurations, predictedOffsets)

def process_poly(targetSequences, args):
    
    targetNotes, targetDurations, targetOffsets = targetSequences
    midiFiles = []
    for i in range(len(targetNotes)):

        generatedStream = stream.Stream()
        generatedStream.append(instrument.Piano())

        # Add notes and durations to stream
        for j in range(len(targetNotes[i])):
            if targetNotes[i][j] != "128":
                pitches = targetNotes[i][j].split(".")
                newChord = chord.Chord()
                for pitch in pitches:
                    newChord.add(int(pitch))
                try:
                    newChord.duration.quarterLength = targetDurations[i][j]
                except:
                    newChord.duration.quarterLength = 0.5
                generatedStream.append(newChord)
            elif targetNotes[i][j] == "128":
                rest = note.Rest()
                try:
                    rest.duration.quarterLength = targetDurations[i][j]
                except:
                    rest.duration.quarterLength = 0.5
                generatedStream.append(rest)

        #generatedStream.show('text')
        midiFiles.append(generatedStream)
    
    notes = [[] for _ in midiFiles]
    durations = [[] for _ in midiFiles]
    offsets = [[] for _ in midiFiles]

    for i, midi in enumerate(midiFiles):
        for element in midi.flat:
            if isinstance(element, note.Note):
                notes[i].append(element.pitch.midi)
                offsets[i].append(element.offset)
                durations[i].append(element.duration.quarterLength)
            elif isinstance(element, chord.Chord):
                for n in element.pitches:
                    notes[i].append(n.midi)
                    offsets[i].append(element.offset)
                    durations[i].append(element.duration.quarterLength)

    return (notes, durations, offsets)


def evaluate(predictions, fileNames, args):
    print("Calculating scores")

    #unpack data
    notePredictions, durPredictions, offsetPredictions = predictions
    numSongs = len(fileNames)
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
#        print("True Notes:")
#        print(trueNotes)
#        print("Predicted Notes:")
#        print(notePredictions[i])
#        print("Predicted Durations")
#        print(durPredictions[i])
        
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
