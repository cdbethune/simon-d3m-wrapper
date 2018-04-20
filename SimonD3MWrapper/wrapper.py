import os.path
import numpy as np
import pandas
import pickle
import requests
import ast
import typing
from json import JSONDecoder
from typing import List

from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = 'Distil'
__version__ = '1.0.0'

Inputs = container.pandas.DataFrame
Outputs = container.List[container.List[str]]

class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    pass

class simon(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d2fa8df2-6517-3c26-bafc-87b701c4043a",
        'version': __version__,
        'name': "simon",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Data Type Predictor'],
        'source': {
            'name': __author__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/simon-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/simon-d3m-wrapper.git@{git_commit}#egg=SimonD3MWrapper'.format(
                git_commit='ddf89ebbd73c2a57483077dd9802c40264612966',
            ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.distil.simon',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_CLEANING,
    })
    
    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
                
        self._decoder = JSONDecoder()
        self._params = {}

    def fit(self) -> None:
        pass
    
    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        pass
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's best guess for the structural type of each input column.
        
        Parameters
        ----------
        inputs : Input pandas frame

        Returns
        -------
        Outputs
            The outputs is two lists of lists, each has length equal to number of columns in input pandas frame. 
            Each entry of the first one is a list of strings corresponding to each column's multi-label classification.
            Each entry of the second one is a list of floats corresponding to prediction probabilities.
        """
        
        """ Accept a pandas data frame, predicts column types in it
        frame: a pandas data frame containing the data to be processed
        -> a list of two lists of lists of 1) column labels and then 2) prediction probabilities
        """
        
        frame = inputs

        try:
            # setup model as you typically would in a Simon main file
            maxlen = 20
            max_cells = 500
            p_threshold = 0.5
        
            DEBUG = True # boolean to specify whether or not print DEBUG information

            checkpoint_dir = "pretrained_models/"

            with open('Categories.txt','r') as f:
                Categories = f.read().splitlines()
    
            # orient the user a bit
            print("fixed categories are: ")
            Categories = sorted(Categories)
            print(Categories)
            category_count = len(Categories)

            execution_config="Base.pkl"

            # load specified execution configuration
            if execution_config is None:
                raise TypeError
            Classifier = Simon(encoder={}) # dummy text classifier
            
            config = Classifier.load_config(execution_config, checkpoint_dir)
            encoder = config['encoder']
            checkpoint = config['checkpoint']

            X = encoder.encodeDataFrame(frame)
            
            # build classifier model    
            model = Classifier.generate_model(maxlen, max_cells, category_count)
            Classifier.load_weights(checkpoint, None, model, checkpoint_dir)
            model_compile = lambda m: m.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['binary_accuracy'])
            model_compile(model)
            y = model.predict(X)
            # discard empty column edge case
            y[np.all(frame.isnull(),axis=0)]=0

            result = encoder.reverse_label_encode(y,p_threshold)

            return result
        except:
            # Should probably do some more sophisticated error logging here
            return "Failed predicting data frame"


if __name__ == '__main__':
    client = simon(hyperparams={})
    # make sure to read dataframe as string!
    # frame = pandas.read_csv("https://query.data.world/s/10k6mmjmeeu0xlw5vt6ajry05",dtype='str')
    frame = pandas.read_csv("https://s3.amazonaws.com/d3m-data/merged_o_data/o_4550_merged.csv",dtype='str')
    result = client.produce(inputs = frame)
    print(result)