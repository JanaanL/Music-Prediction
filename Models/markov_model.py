""" 
This module generates prediction sequences of notes and durations for midi music files
using first order and second order Markov chains.  
"""

import pickle
import glob
import os
import csv
import random
import utils
import numpy as np
import argparse
from music21 import converter, instrument, note, stream, chord, pitch, interval, key

def markov_model():
    args = get_args()
    
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
    
    dicts = (notesToInt, durationToInt)
    scores = process_midi(args)
    probTables = generate_model(scores, dicts, args)
    endElements, primeKeys, primeNoteSets = get_end_elements(dicts, args)

    predictions = predict_sequences(probTables, endElements, dicts, primeKeys, primeNoteSets, args)
    score = evaluate(predictions, args)
    print("The final score is ", score)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1, help="The order level of the Markov Model.  Default is 1 for a first-order Markov Model.")
    parser.add_argument("--data-file", type=str, default="../Data/mono_small/", help="path of data files to load")
    parser.add_argument("--music-type", type=str, default = "mono", choices=["poly", "mono"], help="Type of music that will be used.  Choices includes 'mono' for monophonic music and 'poly' for polyphonic music")
    parser.add_argument("--transpose", type=str, default=None, choices=['all', 'C'], help="Transpose data into all twelve keys for training when option 'all' is specified, and tranpose to either C major or c minor when 'C'  option is specified")
    parser.add_argument("--constrained", default = False, action = "store_true", help="Constrained inference.  Only notes in the prime sequence will be used for the prediction sequence")
    args = parser.parse_args()
    return args

def process_midi(args):
    
    #Load pickled training files
    path = args.data_file + "trainList"
    with open(path, 'rb') as filepath:
        trainFileNames = pickle.load(filepath)
    
    scores = []
    for fname in trainFileNames:
        print("Processing file no: ", fname)
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
    
    return scores


def generate_model(scores, dicts, args):
    print("Generating markov model with order of :", args.level)

    #Unpack data
    notesToInt, durationToInt = dicts
    
    if args.music_type == "mono":
        num_notes = 129
    else:
        num_notes = (len(notesToInt))
    num_durations = len(durationToInt)
    
    if args.level == 1:
        epsilon = 0.001
        notes = np.empty((num_notes, num_notes))
        notes.fill(epsilon)
        durations = np.empty((num_durations, num_durations))
        durations.fill(epsilon)

        firstElement = True
        priorNote = None
        priorDur = None
        for i, song in enumerate(scores):
            if i % 10 == 0:
                print ("Processing song no ", i)
            if args.music_type == "mono":
                for element in song.flat:
                    if isinstance(element, note.Note):
                        if firstElement:
                            priorNote = element.pitch.midi
                            priorDur = durationToInt[element.duration.quarterLength]
                            firstElement = False
                        else:
                            currentNote = element.pitch.midi
                            notes[priorNote][currentNote] +=1
                            currentDur =  durationToInt[element.duration.quarterLength]
                            durations[priorDur][currentDur] +=1
                            priorNote = currentNote
                            priorDur = currentDur
                    elif isinstance(element, note.Rest):
                        if firstElement:
                            priorNote = 128
                            priorDur = durationToInt[element.duration.quarterLength]
                            firstElement = False
                        else:
                            currentNote = 128
                            notes[priorNote][currentNote] +=1
                            currentDur =  durationToInt[element.duration.quarterLength]
                            durations[priorDur][currentDur] +=1
                            priorNote = currentNote
                            priorDur = currentDur

        #music type is "poly"
            else:
                newStream = stream.Stream()
                for part in song.parts:
                    newStream.mergeElements(part)
                for element in newStream.chordify():
                    if isinstance(element, note.Note):
                        if firstElement:
                            priorNote = notesToInt[str(element.pitch.midi)]
                            priorDur = durationToInt[element.duration.quarterLength]
                            firstElement = False
                        else:
                            currentNote = notesToInt[str(element.pitch.midi)]
                            notes[priorNote][currentNote] += 1
                            currentDur = durationToInt[element.duration.quarterLength]
                            durations[priorDur][currentDur] +=1
                            priorNote = currentNote
                            priorDur = currentDur
                    if isinstance(element, note.Rest):
                        if firstElement:
                            priorNote = notesToInt["128"]
                            priorDur = durationToInt[element.duration.quarterLength]
                            firstElement = False
                        else:
                            currentNote = notesToInt["128"]
                            notes[priorNote][currentNote] += 1
                            currentDur = durationToInt[element.duration.quarterLength]
                            durations[priorDur][currentDur] +=1
                            priorNote = currentNote
                            priorDur = currentDur
                    if isinstance(element, chord.Chord):
                        chords = ".".join(str(n.midi) for n in element.pitches)
                        if firstElement:
                            priorNote = notesToInt[chords]
                            priorDur = durationToInt[element.duration.quarterLength]
                            firstElement = False
                        else:
                            currentNote = notesToInt[chords]
                            notes[priorNote][currentNote] += 1
                            currentDur = durationToInt[element.duration.quarterLength]
                            durations[priorDur][currentDur] +=1
                            priorNote = currentNote
                            priorDur = currentDur

    #2nd order markov chain
    else:
        
        epsilon = 0.001
        notes = np.empty((num_notes, num_notes, num_notes))
        notes.fill(epsilon)
        durations = np.empty((num_durations, num_durations, num_durations))
        durations.fill(epsilon)

        # Count notes and durations
        index = 0
        priorNotes = []
        priorDurs = []
        for i, song in enumerate(scores):
            if i % 10 == 0:
                print ("Processing song no ", i)
            if args.music_type == "mono":
                for element in song.flat:
                    if isinstance(element, note.Note):
                        if index < 2:
                            priorNotes.append(element.pitch.midi)
                            priorDurs.append(durationToInt[element.duration.quarterLength])
                            index += 1
                        else:
                            currentNote = element.pitch.midi
                            notes[priorNotes[0]][priorNotes[1]][currentNote] +=1
                            currentDur =  durationToInt[element.duration.quarterLength]
                            durations[priorDurs[0]][priorDurs[1]][currentDur] +=1
                            priorNotes = [priorNotes[1], currentNote]
                            priorDurs = [priorDurs[1], currentDur]
                    elif isinstance(element, note.Rest):
                        if index < 2:
                            priorNotes.append(128)
                            priorDurs.append(durationToInt[element.duration.quarterLength])
                        else:
                            currentNote = 128
                            notes[priorNotes[0]][priorNotes[1]][currentNote] +=1
                            currentDur =  durationToInt[element.duration.quarterLength]
                            durations[priorDurs[0]][priorDurs[1]][currentDur] +=1
                            priorNotes = [priorNotes[1], currentNote]
                            priorDurs = [priorDurs[1], currentDur]

        #music type is "poly"
            else:
                newStream = stream.Stream()
                for part in song.parts:
                    newStream.mergeElements(part)
                for element in newStream.chordify():
                    if isinstance(element, note.Note):
                        if index < 2:
                            priorNotes.append(notesToInt[str(element.pitch.midi)])
                            priorDurs.append(durationToInt[element.duration.quarterLength])
                            index += 1
                        else:
                            currentNote = notesToInt[str(element.pitch.midi)]
                            notes[priorNotes[0]][priorNotes[1]][currentNote] +=1
                            currentDur =  durationToInt[element.duration.quarterLength]
                            durations[priorDurs[0]][priorDurs[1]][currentDur] +=1
                            priorNotes = [priorNotes[1], currentNote]
                            priorDurs = [priorDurs[1], currentDur]

                    if isinstance(element, note.Rest):
                        if firstElement:
                            priorNotes.append(notesToInt["128"])
                            priorDurs.append(durationToInt[element.duration.quarterLength])
                            index += 1
                        else:
                            currentNote = notesToInt["128"]
                            notes[priorNotes[0]][priorNotes[1]][currentNote] +=1
                            currentDur =  durationToInt[element.duration.quarterLength]
                            durations[priorDurs[0]][priorDurs[1]][currentDur] +=1
                            priorNotes = [priorNotes[1], currentNote]
                            priorDurs = [priorDurs[1], currentDur]

                    if isinstance(element, chord.Chord):
                        chords = ".".join(str(n.midi) for n in element.pitches)
                        if firstElement:
                            priorNotes.append(notesToInt[chords])
                            priorDurs.append(durationToInt[element.duration.quarterLength])
                        else:
                            currentNote = notesToInt[chords]
                            notes[priorNote][currentNote] += 1
                            currentDur = durationToInt[element.duration.quarterLength]
                            durations[priorDur][currentDur] +=1
                            priorNote = currentNote
                            priorDur = currentDur

    notes = notes / notes.sum(axis=1, keepdims=True)
    durations = durations / durations.sum(axis=1, keepdims=True)
    
    return (notes, durations)

def get_end_elements(dicts, args):
    
    #Unpack data
    notesToInt, durationToInt = dicts

    #Load pickleld test files
    path = args.data_file + "testList"
    with open(path, 'rb') as filepath:
        testFileNames = pickle.load(filepath)

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

    endNotes = []
    endDurs = []
    primeNoteSets = []

    for i, song in enumerate(primeScores):
        notes = []
        durs = []
        noteSet = set()
        if args.music_type == "mono":
            for element in song.flat:
                if isinstance(element, note.Note):
                    notes.append(element.pitch.midi)
                    durs.append(element.duration.quarterLength)
                    noteSet.add(element.pitch.midi)
                elif isinstance(element, note.Rest):
                    notes.append(128)
                    durs.append(element.duration.quarterLength)
                    noteSet.add(128)
            if args.level == 1:
                endNotes.append(notes[-1])
                endDurs.append(durationToInt[durs[-1]])
            else:
                endNotes.append(notes[-2:])
                durs = durs[-2:]
                durs = [durationToInt[d] for d in durs]
                endDurs.append(durs)
        
        #music type is poly
        else:
            for element in song.flat:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch.midi))
                    durs.append(element.duration.quarterLength)
                    noteSet.add(notesToInt[str(element.pitch.midi)])
                elif isinstance(element, note.Rest):
                    notes.append("128")
                    durs.append(element.duration.quarterLength)
                    noteSet.add(notesToInt["128"])
                elif isinstance(element, chord.Chord):
                    chords = ".".join(str(n.midi) for n in element.pitches)
                    notes.append(chords)
                    durs.append(element.duration.quarterLength)
                    try:
                        noteInt = notesToInt[chords]
                    except:
                        print("Chord not found in dictionary: ", chords)
                        noteInt = notesToInt["128"]
                    noteSet.add(noteInt)

            if args.level == 1:
                try:
                    endNotes.append(notesToInt[notes[-1]])
                except:
                    print("Chord not found in dictionary: ", notes[1])
                    endNotes.append("128")
                endDurs.append(durationToInt[durs[-1]])
            else:
                lastNotes = notes[-2:]
                for k, n in enumerate(lastNotes):
                    try:
                        chordInt = notesToInt[n]
                    except:
                        print("Chord not found in dictionary: ", n)
                        #Replace with rest
                        chordInt = notesToInt["128"]
                    lastNotes[i] = chordInt

                endNotes.append(lastNotes)
                durs = durs[-2:]
                durs = [durationToInt[d] for d in durs]
                endDurs.append(durs)
        primeNoteSets.append(noteSet)

    return (endNotes, endDurs), primeKeys, primeNoteSets

def predict_sequences(probTables, endElements, dicts, primeKeys, primeNoteSets, args):

    #unpack items
    noteProbs, durProbs = probTables
    print(noteProbs)
    endNote, endDur = endElements
    notesToInt, durationToInt = dicts

    #Invert dictionaries
    if args.music_type == "poly":
        intToNotes = {i: c for c, i in notesToInt.items()}
    intToDuration = {i: c for c, i in durationToInt.items()}
    
    predictedNotes = []
    predictedDurs = []
    predictedOffsets = []

    # generate 10 notes
    for i in range(len(endNote)):
        note_seq = []
        dur_seq = []
        offset_seq = []
        priorNote = endNote[i]
        priorDur = endDur[i]
        priorOffset = 0
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        while numBeats <= durationLength:
            if args.level == 1:
                #predNote = np.argmax(noteProbs[priorNote])
                predNote = np.random.choice(noteProbs.shape[0], p=noteProbs[int(priorNote)])
                if args.constrained == True:
                    while predNote not in primeNoteSets[i]:
                        predNote = np.random.choice(noteProbs.shape[0], p=noteProbs[int(priorNote)])

                #predDur = np.argmax(durProbs[priorDur])
                predDur = np.random.choice(durProbs.shape[0], p=durProbs[priorDur])
                priorNote = predNote
                priorDur = predDur
            else:
                #predNote = np.argmax(noteProbs[priorNotes[0]][priorNotes[1]])
                predNote = np.random.choice(noteProbs.shape[0], p=noteProbs[priorNote[0]][priorNote[1]])
                #predDur = np.argmax(durProbs[priorDurs[0]][priorDurs[1]])
                predDur = np.random.choice(durProbs.shape[0], p=durProbs[priorDur[0]][priorDur[1]])
                priorNote = [priorNote[1], predNote]
                priorDur = [priorDur[1], predDur]

            d = intToDuration[predDur]
            offset = priorOffset + d
            note_seq.append(predNote)
            dur_seq.append(predDur)
            offset_seq.append(offset)
            priorOffset = offset
            numBeats += d

        if args.music_type == "poly":
            note_seq = [intToNotes[c] for c in note_seq]
        dur_seq = [intToDuration[c] for c in dur_seq]
        predictedNotes.append(note_seq)
        predictedDurs.append(dur_seq)
        predictedOffsets.append(offset_seq)

    if args.music_type == "poly":
        return create_midis_poly((predictedNotes, predictedDurs, primeKeys), args)
    elif args.music_type == "mono" and  args.transpose == "C":
        return create_midis((predictedNotes, predictedDurs, primeKeys), args)
    else:
        return (predictedNotes, predictedDurs, predictedOffsets)


def evaluate(predictions, args):

    #Unpack data
    notePredictions, durationPredictions, offsetPredictions = predictions

    path = args.data_file + "testList"
    with open(path, 'rb') as filepath:
        testFileNames = pickle.load(filepath)

    numSongs = len(notePredictions)
    totalScores = np.zeros(numSongs)
    totalCScores = np.zeros(numSongs)
    totalPScores = np.zeros(numSongs)

    for i in range(numSongs):
        trueNotes = []
        trueOffsets = []
        true = []
        predictions = []
        file = testFileNames[i].replace(".mid", "")
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
        print("Calculating score for song no ", i)
        print("True Notes:")
        print(trueNotes)
        print("Predicted Notes:")
        print(notePredictions[i])
        
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

    return notes, durations, offsets

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

def predict_sequences_beamsearch(probTables, endElements, dicts, primeKeys, mostCommonNote, args):
    print("Generating notes for predictions using beam search")
    k = args.beam_search #length of beam search
    
    #unpack items
    noteProbs, durProbs = probTables
    endNote, endDur = endElements
    notesToInt, durationToInt = dicts

    #Invert dictionaries
    if args.music_type == "poly":
        intToNotes = {i: c for c, i in notesToInt.items()}
    intToDuration = {i: c for c, i in durationToInt.items()}
    
    predictedNotes = []
    predictedDurs = []
    predictedOffsets = []

    # generate 10 notes
    for i in range(len(endNote)):
        note_seq = []
        dur_seq = []
        offset_seq = []
        priorNote = endNote[i]
        print("The beginning note is ", endNote[i])
        priorDur = endDur[i]
        priorOffset = 0
        durationLength = 10 #10 quarterLength note beats
        numBeats = 0
        
        nBeams = [((), 0) for i in range(k)]
        dBeams = [((), 0) for i in range(k)]

        while numBeats <= durationLength:
            allNBeams = set()
            allDBeams = set()
            beamLen = len(nBeams[0][0])
            for m in range(k):
                if numBeats > 0:
                    priorNote = nBeams[m][0][-1]
                    priorDur = dBeams[m][0][-1]
                bestKNotes = noteProbs[priorNote].argsort()[-k:]
                bestKDurations = durProbs[priorDur].argsort()[-k:]

                for j in range(k):
                    allNBeams.add((
                                tuple(list(nBeams[m][0]) + [bestKNotes[j]]), 
                                noteProbs[priorNote][bestKNotes[j]] + nBeams[m][1]))

                
                    allDBeams.add((
                                tuple(list(dBeams[m][0]) + [bestKDurations[j]]), 
                                durProbs[priorDur][bestKDurations[j]] + dBeams[m][1]))
            
            nBeams = sorted(allNBeams, key=lambda tup:tup[1])[-k:]
            dBeams = sorted(allDBeams, key=lambda tup:tup[1])[-k:]
            bestBeat = dBeams[0][0][-1:][0]
            numBeats += intToDuration[bestBeat]
        
        if args.music_type == "poly":
            nBeams[0][0] = [intToNotes[c] for c in nBeams[0][0]]
        predictedNotes.append(nBeams[0][0])
        dur_seq = [intToDuration[d] for d in dBeams[0][0]]
        predictedDurs.append(dur_seq)
        offset = 0
        for d in dur_seq:
            offset += d
            offset_seq.append(offset)
        predictedOffsets.append(offset_seq)

    if args.transpose ==  "C":
        return create_midis((predictedNotes, predictedDurs, primeKeys), args)
    else:
        return (predictedNotes, predictedDurs, predictedOffsets)


if __name__ == '__main__':
    markov_model()
