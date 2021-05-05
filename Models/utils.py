""" 
This module contains utility functions for the various models in the project
"""
import argparse
import random
import numpy as np
from music21 import stream, interval, pitch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs for training")
    parser.add_argument("--sequence-length", type=int, default=32, help="length of sequences for lstm model")
    parser.add_argument("--checkpoint", type=str, default=None, help="path of checkpoint to resume training")
    parser.add_argument("--initial-epoch", type=int, help="epoch to resume training")
    parser.add_argument("--data-file", type=str, default="../Data/mono_small/", help="path of data files to load")
    parser.add_argument("--transpose", type=str, default=None, choices=['all', 'C'], help="Transpose data into all twelve keys for training when option 'all' is specified, and tranpose to either C major or c minor when 'C'  option is specified")
    parser.add_argument("--name", type=str, default="lstm", help="Name of model")
    parser.add_argument("--embed-dim", type=int, default = 64, help="Number of embedding dimensions")
    parser.add_argument("--music-type", type=str, default = "mono", choices=["poly", "mono"], help="Type of music that will be used.  Choices includes 'mono' for monophonic music and 'poly' for polyphonic music")
    parser.add_argument("--model-type", type=str, default = "single", choices=["single", "parallel"], help="Type of lstm or  encoder/decoder model that will be trained.  Parallel model trains sepearate lstm or encoder/decoder for notes and durations.  Single model trains one lstm or encode/decoder that uses both notes and durations as input.")
    parser.add_argument("--checkpoint-model", type=str, default = "notes", choices=["notes", "durations"], help="Type of model that is checkpointed.  This is in conjuction with the parallel encoder decoder model.  Choices include 'notes' or 'durations'")
    parser.add_argument("--saved-model", type=str, default = "data/models/", help="Path of saved models.  When used for predictiong, this is the path where the saved model will be retrieved.")
    parser.add_argument("--beam-search", type=int, default = None, help="Length of beam search")
    parser.add_argument("--constrained-inference", default=False, action='store_true', help="Predicition will be made with hard constraints.  Specificaly, a note must be in the prime of the song to be considered in the target sequence of the prediction")

    args = parser.parse_args()
    return args

def transposeToAll(midiStream):
    """ 
    This function takes the input stream and transposes it to all keys.  If the input stream
    is in a major key, then it tranposes it to all major keys.  The same is done for minor keys.
    """

    majorKeys = ['A','B-','B','C','C#','D','E-','E','F','F#','G','G#']
    minorKeys = ['a','b-','b','c','c#','d','e-','e','f','f#','g','g#']

    transposedStreams = []
    key = midiStream.analyze('key')
    mode = key.mode
    primaryPitch = key.tonic
    if mode == 'major':
        keys = majorKeys
    elif mode == 'minor':
        keys = minorKeys

    for k in keys:
        i = interval.Interval(primaryPitch, pitch.Pitch(k))
        newKey = midiStream.flat.transpose(i)
        transposedStreams.append(newKey)
    
    return transposedStreams

def transposeToC(midiStream):
    """
    This function takes the input and transposes it to either C major or C minor, depending if the input
    is in a major or minor key.
    """

    key = midiStream.analyze('key')
    mode = key.mode
    primaryPitch = key.tonic
    if mode == 'major':
        i = interval.Interval(primaryPitch, pitch.Pitch('C'))
    elif mode == 'minor':
        i = interval.Interval(primaryPitch, pitch.Pitch('c'))

    transposedStream = midiStream.flat.transpose(i)
    
    return transposedStream
    

def cardinalityScore(true, predictions):
    """
    The function calculates the cardinality score between the gold truth and the prediction.
    The cardinality score attempts to find the best overlap between the true and predicted outputs.
    More information about how this score is calculated can be found at the following website:
    https://www.music-ir.org/mirex/wiki/2020:Patterns_for_Prediction
    """

    T = []

    # calculate transition vectors
    for i in range(len(true)):
        x = true[i][0]
        y = true[i][1]
        for j in range(len(predictions)):
            x_hat = predictions[j][0]
            y_hat = predictions[j][1]
            deltaX = (x - x_hat)
            deltaY = (y - y_hat)
            T.append((deltaX,deltaY))

    # calculate cardinality under each transition vector
    maxCardinality = 0
    for i in range(len(T)):
        t = T[i]
        translatedPreds = []
        
        # calculate translated predictions:
        for j in range(len(predictions)):
            transX = predictions[j][0] + t[0]
            transY = predictions[j][1] + t[1]
            translatedPreds.append((transX,transY))

        # calculate cardinality of translated predictions with truth
        cardinality = 0
        for k in range(len(true)):
            for j in range(len(translatedPreds)):
                if true[k][0] == translatedPreds[j][0] and true[k][1] == translatedPreds[j][1]:
                    cardinality += 1
        if cardinality > maxCardinality:
            maxCardinality = cardinality

    # normalize cardinality score
    score = maxCardinality * 2 / (len(true) + len(predictions))
    return score
    

def pitchScore(truePitches, predPitches):
    """ 
    Calculates the pitchScore between two streams of music by comparing the distance between the two normalized
    histograms of the music streams.
    
    Input:  truePitches -- a list of the pitches in the true output.  The pitches are Midi note number value between 1 and 127.
            predPitches -- a list of the pitches in the predicted output.
    """
    if len(predPitches) == 0:
        return 0

    nMidi = 129 #total number of midi pitches (bins in histographs -- includes 128 which is treated as a rest)
    hist1 = np.zeros(nMidi)
    hist2 = np.zeros(nMidi)

    #create histographs
    for i in range(len(truePitches)):
        pitch = truePitches[i]
        hist1[pitch] += 1
    for i in range(len(predPitches)):
        pitch = predPitches[i]
        hist2[pitch] += 1

    #normalize histographs
    for i in range(nMidi):
        hist1[i] = hist1[i] / len(truePitches)
        hist2[i] = hist2[i] / len(predPitches)

    #calculate distance
    totalDist = 0.0
    for i in range(nMidi):
        totalDist += abs(hist1[i] - hist2[i])

    score = 1 - totalDist / 2.0
    return score

