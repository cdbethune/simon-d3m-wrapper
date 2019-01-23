# Simon D3M Wrapper
Wrapper of the Simon semantic classifier into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater. 

The base library for SIMON can be found here: https://github.com/NewKnowledge/simon

## Install

pip3 install -e git+https://github.com/NewKnowledge/simon-d3m-wrapper.git#egg=SimonD3MWrapper --process-dependency-links

## Output
The output should be a list of two list of lists. 

The first is a list of multilabel label strings (ordered the same as columns).

e.g. if all (except the last-but-one, an 'int') labels are text (remember that the order of labels matches the order in the source file), the output should be as follows:

```[['text'], ['text'], ['int'], ['text']]```

The second is a list of lists of floats (of the same size), providing corresponding confidence probabilities for each label.

## Available Functions

#### produce
Produce primitive's best guess for the structural type of each input column. The input is a pandas dataframe. The output is  a list that has length equal to number of columns in input pandas frame. Each entry is a list of strings corresponding to each column's multi-label classification. Could be empty, signifiying lack of confidence in any classification.
