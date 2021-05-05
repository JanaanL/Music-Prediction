# Music-Prediction

This is my class project for CS6355 Structured Prediction Spring 2021 at the University of Utah.
This project involves music prediction and is structured after the Patterns for Prediction challenge of the 2020 Mirex Challenge.
Details about this challenge can be found here:  https://www.music-ir.org/mirex/wiki/2020:Patterns_for_Prediction.
This project aimed to take an excerpt of music as input and predict a continuation of the excerpt.  The results were scored using the same metrics that were used in the Mirex challenge.  Different music types (monophonic and polyphonic) and input feature representations were used with several learning models to compare the differences in prediction and learning.

The following libraries are needed to run the files in this project:
* Music21
* Keras 2.4.3
* TensorFlow 2.3.1


To prepare the data for this project, run the data.py file.  An example of a command-line argument to run the data file on the monophonic datasets would be:
```
python data.py --music-type "mono"
```

This will split the datasets into training and testing files.  The current ratio is a 90/10 train/test split.  The names of the training and testing files
can be found in the respective data directories as testList and trainList.  These are pickled python lists and are retrieved in the models.  This file will also create appropriate dictionaries for the feature input representations.  For example, for the monophonic datasets, a dictionary will be created that maps unique duration lengths to integer values.  This will be a pickled python dictionary called durationToInt and is found in the respective data directory.  It is retrieved and used in all of the models.  Similar dictionaries can be created for the polyphonic music note input and the combination tuples of notes and durations.  See the file for more details.

## Baselines

The baseline.py file can be used to create some baseline numbers for comparison.  This file will calculate baseline scores for the different datasets.
Three different types of baselines are calculated:  *random, sampled, and sampled with constraints.* 

*Random* baseline sequences are sequences of notes/chords and durations chosen randomly from the training set.
*Sampled* sequences are sequences of notes/chords that are sampled using the distribution of notes/chords from the training set.
*Samples with constraints* sequences are sequences of notes/chords that are sampled using the distribution of notes/chords from the training set.  
However, the sampled note must be in the prime sequence of the song that is predicted.  If the predicted note is not in the prime, another note is sampled 
until one is found that is in the sequence.

An example of a command-line argument to run the baseline file on the monophonic datasets would be:
```
python baselines.py --data-file "../Data/mono_medium/"
```

This project involves three different models:  A Markov chain, an LSTM model, and an LSTM Encoder-Decoder model.

## Markov Chain

The Markov model is either a first-order or second-order Markov chain.  The learned distributions come from the training data. Both the 
monophonic and the polyphonic datasets can be used with the Markov model.  The input data can be transposed to the key of C for better consistency of
the input data.  The output sequence is then transponsed back to the original key.  This is indicated with the `--transpose "C"` command-line argument.  The inference can also be constrained to only include notes that are found in the prime sequence.  This is indicated with the `--constrained` command-line argument.    To indicate a second-order markov chain, use the command-line argument `--level 2`.

An example of a command-line argument to run a first-order Markov model on the monophonic medium dataset using transposition and constrained inference would be:
```
python markov_model.py --data-file "../Data/mono_medium/" --transpose "C" --constrained
```

## LSTM Model

The LSTM model can be run on both the monophonic and the polyphonic datasets.  It can also be run as a multivariate model with notes and durations as two separate inputs, or it can be run as two separate LSTM models, one each for the notes and durations.  For a list of the different options, please look at the help menu for this file.

An example of a command-line argument to run a multivariate LSTM model on the monophonic dataset would be:
```
 python lstm.py --data-file "../Data/mono_medium/" --name "lstm-model1"
 ```

For prediction and scoring the LSTM model, the predict.py file must be used.  Again, there are several inference options that are available.  Please see the help menu for more details.  An example of a command-line argument to predict and score the model above would be:
```
python predict.py --data-file "../Data/mono_medium/" --saved-model "data/models/lstm-model1"
```

There are several variants of the LSTM model that take different input feature representations.  These include a combination feature representation, which is tuples of notes and durations.  The other input feature representation is step sequences of intervals.  This represents the interval differences in the sequence of pitches.

First, to run the combinataion features LSTM model, the data.py file must have been run to create an appropriate dictionary for the combined feature representation.
Please see the data.py help menu for instructions.  An example command-line argument for this LSTM model would be:
```
python lstm_combined.py --data-file "../Data/mono_medium/" --name "lstm-combined"
```
For prediction of this model, the following command-line argument would be:
```
 python predict_combined.py --data-file "../Data/mono_medium/" --saved-model "data/models/lstm-combined"
```

The other LSTM model uses interval changes and durations as the feature inputs.  An example of a command-line argument to use this model would be:
```
python lstm_step.py --data-file "../Data/mono_medium" --name "lstm-step"
```
For prediction using this model, the following command-line argument would be:
```
python predict_step.py --data-file "../Data/mono_medium/" --saved-model "data/models/lstm-step"
```

## LSTM Encoder-Decoder Model

The last model is an LSTM encoder-decoder model.  It can be used on only the monophonic music.  It is a multivariate model with two separate inputs:  notes
and durations.  See the help menu for more options.  A sample command-line argument to run this model on the medium-sized dataset would be:
```
python encoder_model.py --data-file "../Data/mono_medium/" --name "s2s"
```
And for prediction:
```
python predict_encoder.py --data-file "../Data/mono_medium/" --saved-model "data/models/s2s"
```

