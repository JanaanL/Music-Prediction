""" 
This module generates predicted sequences in the testing set using the 
LSTM model.  The predictions are compared to the gold truth and a score is calculated for the accuracy of the predictions.
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
    
    data = get_test_data(testFileNames, notesToInt, durationToInt, args)
    if args.constrained_inference == True:
        noteSets = get_prime_notes(testFileNames, args)

    if args.model_type == "single":
        if args.beam_search:
            targetSequences = predict_sequences_beamsearch(data, notesToInt, durationToInt, args)
        elif args.constrained_inference == True:
            targetSequences = predict_sequences_constrained(data, notesToInt, durationToInt, noteSets, args)
        else:
            targetSequences = predict_sequences(data, notesToInt, durationToInt, args)
    
    #Model type is parallel
    else:
        targetSequences = predict_sequences_parallel(data, notesToInt, durationToInt, args)

    if args.music_type == "poly":
        notes, offsets, durations = create_midis_poly(targetSequences, args)
    else:
        notes, offsets, durations = create_midis(targetSequences, args)
    score = evaluate(notes, offsets, durations, testFileNames, args)

    print("The final score is ", score)


def get_test_data(testFileNames, notesToInt, durationToInt, args):
    
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

    chords = [[] for _ in primeScores]
    notes = [[] for _ in primeScores]
    durations = [[] for _ in primeScores]
    keys = []

    if args.music_type == "mono":

        # Extract notes, durations, offsets and keys
        for i, song in enumerate(primeScores):
            if i % 10 == 0:
                print("Processing file no. ", i)
            
            keys.append(str(song.analyze('key')))
            for element in song.flat:
                if isinstance(element, note.Note):
                    notes[i].append(element.pitch.midi)
                    durations[i].append(element.duration.quarterLength)
#                elif isinstance(element, note.Rest):
#                    notes[i].append(128)
#                    durations[i].append(element.duration.quarterLength)
            notes[i] = notes[i][-sequenceLength:]
            durations[i] = durations[i][-sequenceLength:]
  
    # music type is "poly"
    else:
        # Extract notes, durations, offsets and keys
        for i, song in enumerate(primeScores):
            if i % 10 == 0:
                print("Processing file no. ", i) 
            keys.append(str(song.analyze('key')))
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
            
            notes[i] = notes[i][-sequenceLength:]
            durations[i] = durations[i][-sequenceLength:]
            for j, n in enumerate(notes[i]):
                try:
                    notes[i][j] = notesToInt[n]
                except:
                    print("Chord ", n, " not found in dictionary!")
                    #Didn't find chord in dict, just replace with rest
                    notes[i][j] = notesToInt['128']
    
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
    
    return (notes, durations, keys) 

def get_prime_notes(testFileNames, args):

    primeScores = []

    for fname in testFileNames:
        primeFilePath = os.path.join(args.data_file + "prime_midi/", fname)
        try:
            primeMidi = converter.parse(primeFilePath)
        except:
            print("File would not parse correctly: ", fname)
        
        if args.transpose == "C":
            primeMidi = utils.transposeToC(primeMidi)
        
        primeScores.append(primeMidi)

    noteSets = [set() for _ in primeScores]

    for i, song in enumerate(primeScores):
        for element in song.flat:
            if isinstance(element, note.Note):
                noteSets[i].add(element.pitch.midi)
    return noteSets

def predict_sequences(data, notesToInt, durationToInt, args):
    print("Predicting sequences...")

    modelPath = args.saved_model
    sequenceLength = args.sequence_length
    model = load_model(modelPath)
    
    #unpack data
    primeNotes, primeDurations, keys = data

    num_notes = 130
    #Invert dictionaries
    if args.music_type == "poly":
        intToNotes = {i: c for c, i in notesToInt.items()}
        num_notes = len(notesToInt)
    intToDuration = {i: c for c, i in durationToInt.items()}
    num_durations = len(durationToInt)

    targetNotes = []
    targetDurations = []
    
    def predictions(noteSequence, durationSequence):
        notes, durations = model.predict([noteSequence, durationSequence])
        
#        if args.sampling:
#            predNote = np.random.choice(129, p=notes[0])
#            predDuration = np.random.choice(num_durations, p=durations[0])
#            return predNote, predDuration
#        else:

        return np.argmax(notes[0]), np.argmax(durations[0])

    # generate 10 notes
    for i in range(len(primeNotes)):
        note_seq = []
        dur_seq = []
        noteInput = np.expand_dims(primeNotes[i],0)
        durationInput = np.expand_dims(primeDurations[i],0)
    
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        while numBeats <= durationLength:
        #for note_index in range(10):
            predNote, predDuration = predictions(noteInput, durationInput)
            note_seq.append(predNote)
            dur_seq.append(predDuration)
            noteInput[0][:-1] = noteInput[0][1:]
            noteInput[0][-1] = predNote
            durationInput[0][:-1] = durationInput[0][1:]
            durationInput[0][-1] = predDuration
            numBeats += intToDuration[predDuration]

        
        if args.music_type == "poly":
            note_seq = [intToNotes[c] for c in note_seq]
        dur_seq = [intToDuration[c] for c in dur_seq]
        targetNotes.append(note_seq)
        targetDurations.append(dur_seq)

    return targetNotes, targetDurations, keys

def predict_sequences_constrained(data, notesToInt, durationToInt, noteSets, args):
    print("Predicting constrained sequences...")

    modelPath = args.saved_model
    sequenceLength = args.sequence_length
    model = load_model(modelPath)
    
    #unpack data
    primeNotes, primeDurations, keys = data

    num_notes = 130
    #Invert dictionaries
    intToDuration = {i: c for c, i in durationToInt.items()}
    num_durations = len(durationToInt)

    targetNotes = []
    targetDurations = []
    
    def predictions(noteSequence, durationSequence, inputNotes):
        notes, durations = model.predict([noteSequence, durationSequence])
                
        predNote = np.argmax(notes[0])
        while predNote not in inputNotes:
            notes[0][predNote] = float('-inf')
            predNote = np.argmax(notes[0])
        return predNote, np.argmax(durations[0])

    # generate 10 notes
    for i in range(len(primeNotes)):
        note_seq = []
        dur_seq = []
        noteInput = np.expand_dims(primeNotes[i],0)
        durationInput = np.expand_dims(primeDurations[i],0)
    
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        while numBeats <= durationLength:
        #for note_index in range(10):
            predNote, predDuration = predictions(noteInput, durationInput, noteSets[i])
            note_seq.append(predNote)
            dur_seq.append(predDuration)
            noteInput[0][:-1] = noteInput[0][1:]
            noteInput[0][-1] = predNote
            durationInput[0][:-1] = durationInput[0][1:]
            durationInput[0][-1] = predDuration
            numBeats += intToDuration[predDuration]
        
        if args.music_type == "poly":
            note_seq = [intToNotes[c] for c in note_seq]
        dur_seq = [intToDuration[c] for c in dur_seq]
        targetNotes.append(note_seq)
        targetDurations.append(dur_seq)

    return targetNotes, targetDurations, keys

def predict_sequences_parallel(data, notesToInt, durationToInt, args):
    print("Predicting sequences...")

    sequenceLength = args.sequence_length
    
    #unpack data
    primeNotes, primeDurations, keys = data

    #Invert duration dictionaries
    num_notes = 130
    if args.music_type == "poly":
        intToNotes = {i: c for c, i in notesToInt.items()}
        num_notes = len(notesToInt)
    intToDuration = {i: c for c, i in durationToInt.items()}
    num_durations = len(durationToInt)
    print('The number of unique durations is ', num_durations)

    targetNotes = []
    targetDurations = []
    
    # generate 10 notes
    for (dataType, prime_inputs) in zip(["durations", "notes"], [primeDurations, primeNotes]):
        if dataType == "durations":
            modelPath = "tmp/checkpoints/lstm/7/durations/weights.49.hdf5"
        else:
            modelPath = "data/models/lstm7-notes"
        #modelPath = args.saved_model + "-" + dataType
        model = load_model(modelPath)
        for i in range(len(prime_inputs)):
            output_seq = []
            input_seq = np.expand_dims(prime_inputs[i],0)
            durationLength = 10 #10 quarterLength note beats
            numBeats = 0

            if dataType == "durations":
                totalSteps = 0
            while numBeats <= durationLength:
            #for note_index in range(10):
                pred = model.predict(input_seq)
                pred = np.argmax(pred)
                output_seq.append(pred)
                input_seq[0][:-1] = input_seq[0][1:]
                input_seq[0][-1] = pred
                if dataType == "durations":
                    d = intToDuration[pred]
                    if isinstance(d, str):
                        d = 0
                    numBeats += d
                    totalSteps += 1
                else:
                    numBeats += 1
                    durationLength = totalSteps
            
            if dataType == "notes":
                if args.music_type == "poly":
                    output_seq = [intToNotes[c] for c in output_seq]
                targetNotes.append(output_seq)
            else:
                output_seq = [intToDuration[c] for c in output_seq]
                targetDurations.append(output_seq)

    return targetNotes, targetDurations, keys

def create_midis(targetSequences, args):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
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

def create_midis_poly(targetSequences, args):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    print("Creating polyphonic midi files for predicted sequences")
    targetNotes, targetDurations, keys = targetSequences
    midiFiles = []
    for i in range(len(targetNotes)):

        if i % 10 == 0:
            print("Creating midi file for song no. ", i)
        
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
            elif isinstance(element, chord.Chord):
                for n in element.pitches:
                    notes[i].append(n.midi)
                    offsets[i].append(element.offset)
                    durations[i].append(element.duration.quarterLength)

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

def predict_sequences_beamsearch(data, notesToInt, durationToInt, args):

    print("Generating notes for predictions using beam search")
    k = args.beam_search #length of beam search
    modelPath = args.saved_model
    model = load_model(modelPath)

    #unpack data
    primeNotes, primeDurations, keys = data
    
    num_notes = 130
    #Invert dictionaries
    if args.music_type == "poly":
        intToNotes = {i: c for c, i in notesToInt.items()}
        num_notes = len(notesToInt)
    intToDuration = {i: c for c, i in durationToInt.items()}
    num_durations = len(durationToInt)

    targetNotes = []
    targetDurations = []

    sequenceLength = args.sequence_length
    
    # generate 10 notes

    for i in range(len(primeNotes)):
        
        durationProbs = []
        noteInput = np.expand_dims(primeNotes[i],0)
        durationInput = np.expand_dims(primeDurations[i],0)
    
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        nBeams = [((), 0) for i in range(k)]

        dBeams = [((), 0) for i in range(k)]
        nInput = copy.deepcopy(noteInput)
        dInput = copy.deepcopy(durationInput)
        while numBeats <= durationLength:
            allNBeams = set()
            allDBeams = set()
            beamLen = len(nBeams[0][0])
            #print("The beam length is ", beamLen)
            for m in range(k):
                if beamLen > 0 and beamLen <= sequenceLength:
                    nInput[0][:-beamLen] = noteInput[0][beamLen:]
                    nInput[0][-beamLen:] = list(nBeams[m][0])
                    dInput[0][:-beamLen] = durationInput[0][beamLen:]
                    dInput[0][-beamLen:] = list(dBeams[m][0])
                elif beamLen > sequenceLength:
                    nInput = np.expand_dims(list(nBeams[m][0])[-sequenceLength:], 0)
                    dInput = np.expand_dims(list(dBeams[m][0])[-sequenceLength:], 0)

                notes, durations = model.predict([nInput, dInput])
                bestKNotes = notes[0].argsort()[-k:]
                bestKDurations = durations[0].argsort()[-k:]
                for j in range(k):
                    allNBeams.add((
                                tuple(list(nBeams[m][0]) + [bestKNotes[j]]), 
                                np.log(notes[0][bestKNotes[j]]) + nBeams[m][1]))

                
                    allDBeams.add((
                                tuple(list(dBeams[m][0]) + [bestKDurations[j]]), 
                                np.log(durations[0][bestKDurations[j]]) + dBeams[m][1]))
            nBeams = sorted(allNBeams, key=lambda tup:tup[1])[-k:]
            dBeams = sorted(allDBeams, key=lambda tup:tup[1])[-k:]
            bestBeat = dBeams[0][0][-1:][0]
            numBeats += intToDuration[bestBeat]
        
        if args.music_type == "poly":
            nBeams[0][0] = [intToNotes[c] for c in nBeams[0][0]]
        targetNotes.append(nBeams[0][0])
        dur_seq = [intToDuration[d] for d in dBeams[0][0]]
        targetDurations.append(dur_seq)

    return targetNotes, targetDurations, keys


if __name__ == '__main__':
    generate()
