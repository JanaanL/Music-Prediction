""" 
This module generates sequences in the testing set using the 
LSTM model with the intervalfeature representation.  This feature  is a sequence of intervals that represent the differences in pitch values for the note input sequence.  The predictions are compared to the gold truth and a score is calculated for the accuracy of the predictions.
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
    
    path = args.data_file + "durationToInt"
    with open(path, 'rb') as filepath:
        durationToInt = pickle.load(filepath)
        print("The number of unique durations used for this data is: ", len(durationToInt))
    
    data = get_test_data(testFileNames, durationToInt, args)
    if args.constrained_inference == True:
        noteSets = get_prime_notes(testFileNames, args)

    if args.model_type == "single":
        if args.beam_search:
            targetSequences = predict_sequences_beamsearch(data, notesToInt, durationToInt, args)
        elif args.constrained_inference == True:
            targetSequences = predict_sequences_constrained(data, notesToInt, durationToInt, noteSets, args)
        else:
            targetSequences = predict_sequences(data, durationToInt, args)
    
    #Model type is parallel
    else:
        targetSequences = predict_sequences_parallel(data, notesToInt, durationToInt, args)

    if args.music_type == "poly":
        notes, offsets, durations = create_midis_poly(targetSequences, args)
    else:
        notes, offsets, durations = create_midis(targetSequences, args)
    score = evaluate(notes, offsets, durations, testFileNames, args)

    print("The final score is ", score)


def get_test_data(testFileNames, durationToInt, args):
    
    sequenceLength = args.sequence_length
    primeScores = []
    primeKeys = []

    for fname in testFileNames:
        primeFilePath = os.path.join(args.data_file + "prime_midi/", fname)
        #root = os.path.splitext(os.path.basename(f))
        try:
            primeMidi = converter.parse(primeFilePath)
        except:
            print("File would not parse correctly: ", fname)
        
        if args.transpose == "C":
            primeKeys.append(primeMidi.analyze('key'))
            primeMidi = utils.transposeToC(primeMidi)
        
        primeScores.append(primeMidi)

    steps = [[] for _ in primeScores]
    notes = [[] for _ in primeScores]
    durations = [[] for _ in primeScores]
    keys = []

    firstElement = True
    # Extract notes, durations, offsets and keys
    for i, song in enumerate(primeScores):
        if i % 10 == 0:
            print("Processing file no. ", i)
        
        keys.append(str(song.analyze('key')))
        for element in song.flat:
            if isinstance(element, note.Note):
                if not firstElement:
                    diff = element.pitch.midi - priorNote + 130
                    steps[i].append(diff)
                    notes[i].append(element.pitch.midi)
                    durations[i].append(element.duration.quarterLength)
                else:
                    firstElement = False
                priorNote = element.pitch.midi

            elif isinstance(element, note.Rest):
                if not firstElement:
                    steps[i].append(130)
                    durations[i].append(element.duration.quarterLength)
        
        steps[i] = steps[i][-sequenceLength:]
        durations[i] = durations[i][-sequenceLength:]
        notes[i] = notes[i][-1]

    for j, durSeq in enumerate(durations):
        for k, dur in enumerate(durSeq):
            try:
                durations[j][k] = durationToInt[dur]
            except:
                print("Duration ", dur, " not found in dictionary!")
                #Didn't find duration in dict, just use 0.5
                durations[j][k] = durationToInt[0.5]

    if args.transpose == "C":
        keys = primeKeys
    
    return (steps, notes, durations, keys) 

def predict_sequences(data, durationToInt, args):
    print("Predicting sequences...")

    modelPath = args.saved_model
    sequenceLength = args.sequence_length
    model = load_model(modelPath)
    
    #unpack data
    primeSteps, lastNotes, primeDurations, keys = data
    
    num_steps = 260
    #Invert dictionaries
    intToDuration = {i: c for c, i in durationToInt.items()}
    num_durations = len(durationToInt)

    targetNotes = []
    targetDurations = []
    
    def predictions(stepSequence, durationSequence):
        steps, durations = model.predict([stepSequence, durationSequence])
        
#        if args.sampling:
#            predNote = np.random.choice(129, p=notes[0])
#            predDuration = np.random.choice(num_durations, p=durations[0])
#            return predNote, predDuration
#        else:

        return np.argmax(steps[0]), np.argmax(durations[0])

    # generate 10 notes
    for i in range(len(primeSteps)):
        note_seq = []
        dur_seq = []
        priorNote = lastNotes[i]
        stepInput = np.expand_dims(primeSteps[i],0)
        durationInput = np.expand_dims(primeDurations[i],0)
    
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        while numBeats <= durationLength:
        #for note_index in range(10):
            predStep, predDuration = predictions(stepInput, durationInput)
            predNote = priorNote + (predStep - 130)
            note_seq.append(predNote)
            priorNote = predNote
            stepInput[0][:-1] = stepInput[0][1:]
            stepInput[0][-1] = predStep
            durationInput[0][:-1] = durationInput[0][1:]
            durationInput[0][-1] = predDuration
            try:
                d = intToDuration[predDuration]
            except:
                d = 0.0
            numBeats += d
            dur_seq.append(d)

        
        targetNotes.append(note_seq)
        targetDurations.append(dur_seq)
    return targetNotes, targetDurations, keys

def create_midis(targetSequences, args):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    print("Creating midi files for predicted sequences")
    targetNotes, targetDurations, keys = targetSequences
    midiFiles = []
    for i in range(len(targetNotes)):

        if i % 10 == 0:
            print("Creating midi file for song no. ", i)
        
        generatedStream = stream.Stream()
        generatedStream.append(instrument.Piano())

        # Add notes and durations to stream
        for j in range(len(targetNotes[i])):
            if targetNotes[i][j] != 128:
                newNote = note.Note(targetNotes[i][j])
                try:
                    newNote.duration.quarterLength = targetDurations[i][j]
                except:
                    newNote.duration.quarterLength = 0.5
                generatedStream.append(newNote)
            elif targetNotes[i][j] == 128:
                rest = note.Rest()
                try:
                    rest.duration.quarterLength = targetDurations[i][j]
                except:
                    rest.duration.quarterLength = 0.5
                generatedStream.append(rest)

        if args.transpose == "C":
            mode = keys[i].mode
            if mode == 'major':
                i = interval.Interval(pitch.Pitch('C'), keys[i].tonic)
            elif mode == 'minor':
                i = interval.Interval(pitch.Pitch('c'), keys[i].tonic)

            generatedStream = generatedStream.flat.transpose(i)

        #generatedStream.show('text')
        midiFiles.append(generatedStream)
    
    notes = [[] for _ in midiFiles]
    offsets = [[] for _ in midiFiles]
    durations = [[] for _ in midiFiles]

    for i, midi in enumerate(midiFiles):
        for element in midi.flat:
            if isinstance(element, note.Note):
                notes[i].append(element.pitch.midi)
                offsets[i].append(element.offset)
                durations[i].append(element.duration.quarterLength)

    return notes, offsets, durations


def evaluate(notePredictions, offsetPredictions, durationPredictions, fileNames, args):
    print("Calculating scores")
    """
    Evaluates the accuracy of the prediction using a cardinality score and a pitch score.
    Each of the two scores is weighted equally, and the final score is the total.
    The total maximum score is 1.0.
    """

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
        totalCScores[i] = cScore * 0.5
        totalPScores[i] = pScore * 0.5
        totalScores[i] = totalScore

    print("Total Cardinality Score is: ", np.sum(totalCScores) / numSongs)
    print("Total Pitch Score is: ", np.sum(totalPScores) / numSongs)
    
    return np.sum(totalScores) / numSongs
if __name__ == '__main__':
    generate()
