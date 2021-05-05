""" This module prepares midi file data, separating training and testing filenames and 
creates the dictionaries for the different data sizes and datasets"""
import glob
import pickle
import numpy as np
import os
import math
import random
import utils
import predict
import argparse
from music21 import converter, note, duration, chord, stream, interval, pitch, chord

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--music-type", type=stre, default="mono", choices=["mono","poly"], help="Type of music that will be processed.  Choices include 'mono' for monophonic music and 'poly' for polyphonic music")
    parser.add_argument("--dict-type", type=str, default="single", choices=["single","combination"], help="Type of dictionary to be created.  Choices include 'single' for either a single note or duration dictionary or 'combination' for a tuple combination of notes and durations.")
    args = pareser.parse_args()
    return args


def split_data(datasets, split=0.1):

    for dataPath in datasets:
        print("Partitioning data in the directory: ", dataPath)
        fileNames = []
        for file in glob.glob(dataPath + "prime_midi/*.mid"):
            fname = os.path.basename(file)
            fileNames.append(fname)
    
        #split data into training and testing
        num_list = random.sample(range(0, len(fileNames)), int(len(fileNames)*split))
        trainList = [fileNames[i] for i in range(len(fileNames)) if i not in num_list]
        testList = [fileNames[i] for i in num_list]

        #save train and test file names
        with open(dataPath + 'trainList', 'wb') as filepath:
            pickle.dump(trainList, filepath)
        with open(dataPath + 'testList', 'wb') as filepath:
            pickle.dump(testList, filepath)

    print("Finished partiioning datasets")
    
def print_filenames(datasets):

    for dataPath in datasets:
        with open(dataPath + 'trainList', 'rb') as filepath:
            trainList = pickle.load(filepath)
            print("The train list for ", dataPath, " : ")
            print(trainList)
            print()
        with open(dataPath + 'testList', 'rb') as filepath:
            testList = pickle.load(filepath)
            print("The test list for ", dataPath, " : ")
            print(testList)
            print()

def create_dicts(datasets, args):
    """
    Creates dictionaries for input.  
    Two different types of datasets: ["mono", "poly"]
    Two diferent types of dictionary types: ["single", "combined"]

    For "mono" music, the durations are sampled from the training data and then each unique duration is mapped
    to an integer value in the dictionary.

    For "poly" music, the chords, which are composed of strings of pitches (e.g. "70.65.64"), are sampled from
    the training data and then each unique chord is mapped to an integer value in the dictionary.
    The durations dictionary is also created and is similar to the duration dictionary for the monophonic datasets.

    When the dictionary type is "combined", a tuples of pitch and duration are sampled from the training data.
    Each unique tuple is then mapped to an integer value in the dictionary.

    The dictionaries are saved in the dataset filepath.
    """

    for dataPath in datasets:
        print("Parsing data files at ", dataPath)
        with open(dataPath + 'trainList', 'rb') as filepath:
            trainList = pickle.load(filepath)
    
        primeScores = []
        for i, fname in enumerate(trainList):
            if i % 10 == 0:
                print("parsing file no ", i)
            filePath = os.path.join(dataPath + "prime_midi/",fname)
            try:
                midi_in = converter.parse(filePath)
                midi_out = converter.parse(os.path.join(dataPath + "cont_true_midi/", fname))
                midi_in.append(midi_out)
                primeScores.append(midi_in)
        
            except:
                print("File would not parse correctly: ", fname)
            
        if args.music_type == "mono":
            if args.dict_type == "single":
                durations = set()
                # Extract notes, chords, durations, and keys
                for i, song in enumerate(primeScores):
                    for element in song.flat:
                        if isinstance(element, note.Note) or isinstance(element, note.Rest):
                            durations.add(element.duration.quarterLength)
                #Add beginning and ending characters to durations (used for encoder-decoder model")
                #durations.add("B")
                #durations.add("E")
                print("The number of unique durations is ", len(durations))

                durationToInt = dict(zip(durations, list(range(0, len(durations)))))
                with open(dataPath + "durationToInt", "wb") as filepath:
                    pickle.dump(durationToInt, filepath)
                print("Saving duration diction to ", dataPath)
            
            elif args.dict_type == "combination":
                notesAndDurations = set()        
                
                # Extract notes, chords, durations, and keys
                for i, song in enumerate(primeScores):
                    for element in song.flat:
                        if isinstance(element, note.Note):
                            notesAndDurations.add((element.pitch.midi, element.duration.quarterLength))
                        elif isinstance(element, note.Rest):
                            notesAndDurations.add((128, element.duration.quarterLength))
            
            #Add beginning and ending characters to durations (used for encoder-decoder model")
                #notesAndDurations.add((0,0))  #start sequence
                #notesAndDurations.add((129,0))  #end sequence
                noteDurToInt = dict(zip(notesAndDurations, list(range(0, len(notesAndDurations)))))

                with open(dataPath + "noteDurToInt", "wb") as filepath:
                    pickle.dump(noteDurToInt, filepath)
                print("Saving notes and durations dictionary to ", dataPath)
                print("The length of the dictionary is ", len(noteDurToInt))
        
        
        #Data Type is Polyphonic -- Parse Chords
        else:
            if args.dict_type == "single":
                notes = set()
                durations = set()
                # Extract notes, chords, durations, and keys
                for i, song in enumerate(primeScores):
                    if i % 10 == 0:
                        print("Parsing chords and notes in file no. ", i)
                    if transpose:
                        song = utils.transposeToC(song)
                    newStream = stream.Stream()
                    for part in song.parts:
                        newStream.mergeElements(part)
                    for element in newStream.chordify():
                        if isinstance(element, note.Note):
                            notes.add(str(element.pitch.midi))
                            durations.add(element.duration.quarterLength)
                        elif isinstance(element, note.Rest):
                            notes.add("128")
                            durations.add(element.duration.quarterLength)
                        elif isinstance(element, chord.Chord):
                            chords = ".".join(str(n.midi) for n in element.pitches) 
                            notes.add(chords)
                            durations.add(element.duration.quarterLength)

                #Add beginning and ending characters to durations (used for encoder-decoder model")
                #notes.add("128")
                #durations.add("B")
                #durations.add("E")
                print("The number of unique notes/chords is ", len(notes))
                print("The number of unique durations is ", len(durations))
                notesToInt = dict(zip(notes, list(range(0, len(notes)))))
                durationToInt = dict(zip(durations, list(range(0, len(durations)))))
                with open(dataPath + "notesToInt", "wb") as filepath:
                    pickle.dump(notesToInt, filepath)
                with open(dataPath + "durationToInt", "wb") as filepath:
                    pickle.dump(durationToInt, filepath)

                print("Saving duration dictionary to ", dataPath)
            
            #dict type is "combination"
            else:
                notesAndDurations = set()        
                # Extract notes, chords, durations, and keys
                for i, song in enumerate(primeScores):
                    for element in song.chordify():
                        if isinstance(element, note.Note):
                            notesAndDurations.add((str(element.pitch.midi), element.duration.quarterLength))
                        elif isinstance(element, note.Rest):
                            notesAndDurations.add((128, element.duration.quarterLength))
                        elif isinstance(element, chord.Chord):
                            chords = ".".join(str(n.midi) for n in element.pitches) 
                            notesAndDurations.add((chords, element.duration.quarterLength))
            
                #Add beginning and ending characters to durations (used for encoder-decoder model")
                #notesAndDurations.add((0,0))  #start sequence
                #notesAndDurations.add((129,0))  #end sequence
                noteDurToInt = dict(zip(notesAndDurations, list(range(0, len(notesAndDurations)))))
                print("The number of unique note/dur combinations are :", len(notesAndDurations))
                
                with open(dataPath + "noteDurToInt", "wb") as filepath:
                    pickle.dump(noteDurToInt, filepath)
                print("Saving notes and durations dictionary to ", dataPath)

def test_true_scores(dataPath):
    """
    Function that tests the scores on the true data (as a sanity check)
    """
    with open(dataPath + 'testList', 'rb') as filepath:
        testList = pickle.load(filepath)
    
    midiFiles = []
    for i, fname in enumerate(testList):
        targetFilePath = os.path.join(dataPath + "cont_true_midi/",fname)
        try:
            midi = converter.parse(targetFilePath)
            midiFiles.append(midi.flat)

        except:
            print("File would not parse correctly: ", fname)
        
    notes = [[] for _ in midiFiles]
    offsets = [[] for _ in midiFiles]
    durations = [[] for _ in midiFiles]

    for i, midi in enumerate(midiFiles):
        for element in midi.flat:
            if isinstance(element, note.Note):
                notes[i].append(element.pitch.midi)
                offsets[i].append(element.offset)
                durations[i].append(element.duration.quarterLength)

    score = predict.evaluate(notes, offsets, durations, testList, args)

    print("The final score is ", score)
        

if __name__ == '__main__':
    args = get_args()
    if args.music_type == "mono":
        small = "../Data/mono_small/"
        medium = "../Data/mono_medium/"
        #large = "../Data/mono_large/"
    else:
        small = "../Data/poly_small/"
        medium = "../Data/poly_medium/"
        #large = "../Data/poly_large/"
    
    datasets = [small, medium]
    split_data(datasets)
    print_filenames(datasets)
    create_dicts(datasets, args)
    #test_true_scores(small)

    
