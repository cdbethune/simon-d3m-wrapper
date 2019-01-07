import os.path
import numpy as np
import pandas
import typing

from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *

from Simon.penny.guesser import guess

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base
from d3m.primitives.datasets import DatasetToDataFrame

from common_primitives import utils as utils_cp

__author__ = 'Distil'
__version__ = '1.2.1'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    overwrite = hyperparams.UniformBool(default = False, semantic_types = [
        'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='whether to overwrite manual annotations with SIMON annotations')
    pass

class simon(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d2fa8df2-6517-3c26-bafc-87b701c4043a",
        'version': __version__,
        'name': "simon",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Data Type Predictor','Semantic Classification','Text','NLP','Tabular'],
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
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        },
            {
            "type": "TGZ",
            "key": "simon_models",
            "file_uri": "http://public.datadrivendiscovery.org/simon_models.tar.gz",
            "file_digest":"c0e112493d796f472f5fe35087eac695d2845ace08b1fe825a0a0328caaf9dfc"
        },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.distil.simon',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_CLEANING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)

        self.volumes = volumes

    def _produce_annotations(self, inputs: Inputs) -> Outputs:
        """
        Private method that produces primtive's best guess for structural type of each input column

        Parameters
        ----------
        inputs: Input pandas frame

        Returns
        -------
        Outputs
            The outputs is two lists of lists, each has length equal to number of columns in input pandas frame.
            Each entry of the first one is a list of strings corresponding to each column's multi-label classification.
            Each entry of the second one is a list of floats corresponding to prediction probabilities.
        """
        frame = inputs

        # setup model as you typically would in a Simon main file
        maxlen = 20
        max_cells = 500
        p_threshold = 0.5

        DEBUG = True # boolean to specify whether or not print DEBUG information

        checkpoint_dir = self.volumes["simon_models"]+"/pretrained_models/"

        with open(self.volumes["simon_models"]+"/Categories.txt",'r') as f:
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
        model_compile = lambda m: m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['binary_accuracy'])
        model_compile(model)
        y = model.predict(X)
        # discard empty column edge case
        y[np.all(frame.isnull(),axis=0)]=0

        result = encoder.reverse_label_encode(y,p_threshold)

        ## LABEL COMBINED DATA AS CATEGORICAL/ORDINAL
        print("Beginning Guessing categorical/ordinal classifications...")
        category_count = 0
        ordinal_count = 0
        raw_data = frame.as_matrix()
        for i in np.arange(raw_data.shape[1]):
            tmp = guess(raw_data[:,i], for_types ='category')
            if tmp[0]=='category':
                category_count += 1
                tmp2 = list(result[0][i])
                tmp2.append('categorical')
                result[0][i] = tuple(tmp2)
                result[1][i].append(1)
                if ('int' in result[1][i]) or ('float' in result[1][i]) \
                    or ('datetime' in result[1][i]):
                        ordinal_count += 1
                        tmp2 = list(result[0][i])
                        tmp2.append('ordinal')
                        result[0][i] = tuple(tmp2)
                        result[1][i].append(1)
        print("Done with statistical variable guessing")
        ## FINISHED LABELING COMBINED DATA AS CATEGORICAL/ORDINAL

        Classifier.clear_session()

        out_df = pandas.DataFrame.from_records(list(result)).T
        out_df.columns = ['semantic types','probabilities']
        return(out_df)

    def produce_metafeatures(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's best guess for the structural type of each input column.

        Parameters
        ----------
        inputs : D3M Dataset object

        Returns
        -------
        Outputs
            The outputs is two lists of lists, each has length equal to number of columns in input pandas frame.
            Each entry of the first one is a list of strings corresponding to each column's multi-label classification.
            Each entry of the second one is a list of floats corresponding to prediction probabilities.
        """

        simon_df = self._produce_annotations(inputs)

        # add metadata to output data frame
        #simon_df = d3m_DataFrame(out_df)
        # first column ('semantic types')
        col_dict = dict(simon_df.metadata.query((metadata_base.All_ELEMENTS, 0)))
        col_dict['structural_type'] = type("this is text")
        col_dict['name'] = 'semantic types'
        col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        simon_df.metadata = simon_df.metadata.update((metadata_base.All_ELEMENTS, 0), col_dict)
        # second column ('probabilities')
        col_dict = dict(simon_df.metadata.query((metadata_base.All_ELEMENTS, 1)))
        col_dict['structural_type'] = type("this is text")
        col_dict['name'] = 'probabilities'
        col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        simon_df.metadata = simon_df.metadata.update((metadata_base.All_ELEMENTS, 1), col_dict)

        return CallResult(simon_df)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> None: #CallResult[Outputs]:
        """
        Add SIMON annotations if manual annotations do not exist. Hyperparameter overwrite controls whether manual 
        annotations should be overwritten with SIMON annotations.

        Parameters
        ----------
        inputs : Input pandas frame

        Returns
        -------
        Outputs
            Input pandas frame with metadata augmented and optionally overwritten
        """
        # calculate SIMON annotations
        simon_annotations = self._produce_annotations(inputs)
        overwrite = self.hyperparams['overwrite']

        # overwrite or augment metadata with SIMON annotations
        for i in range(0, inputs.shape[1]):
            metadata = inputs.metadata.query_column(i)
            col_dict = dict(metadata)
            structural_type = metadata['structural_type']

            # structural types
            if overwrite or structural_type is "" or structural_type is None or 'structural_type' not in metadata.keys():
                col_dict['structural_type'] = type("string")

            # semantic types
            # overwrite with SIMON annotation of highest probability
            # TODO: add functionality to sample from probabilities??
            index = simon_annotations['probabilities'][i].index(max(simon_annotations['probabilities'][i]))
            annotation = simon_annotations['semantic types'][i][index]
            semantic_types = metadata['semantic_types']
            if overwrite or semantic_types is "" or semantic_types is None or 'semantic_types' not in metadata.keys():           
                if annotation == 'categorical':
                    col_dict['semantic_types'][0] = 'https://metadata.datadrivendiscovery.org/types/CategoricalData'
                elif annotation == 'address' or annotation == 'email' or annotation == 'text' or annotation == 'uri':
                    col_dict['semantic_types'][0] = 'https://schema.org/Text'
                elif annotation == 'boolean':
                    col_dict['semantic_types'][0] = 'https://schema.org/Boolean'
                elif annotation == 'datetime':
                    col_dict['semantic_types'][0] = 'https://schema.org/DateTime'
                elif annotation == 'float':
                    col_dict['semantic_types'][0] = 'https://schema.org/Float'
                elif annotation == 'int':
                    col_dict['semantic_types'][0] = 'https://schema.org/Integer'
                elif annotation == 'phone':
                    col_dict['semantic_types'][0] = 'https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber'
                elif annotation == 'ordinal':
                    col_dict['semantic_types'][0] = 'https://metadata.datadrivendiscovery.org/types/OrdinalData'
            if semantic_types is "" or semantic_types is None or 'semantic_types' not in metadata.keys():
                col_dict['semantic_types'][1] = 'https://metadata.datadrivendiscovery.org/types/Attribute'
            inputs.metadata = inputs.metadata.update_column(i, col_dict)
        print(inputs.metadata.query_column(0))
        return CallResult(inputs)

if __name__ == '__main__':  
    # LOAD DATA AND PREPROCESSING
    input_dataset = container.Dataset.load("file:///home/196_autoMpg/TRAIN/dataset_TRAIN/datasetDoc.json")
    ds2df_client = DatasetToDataFrame(hyperparams={"dataframe_resource":"0"})
    df = ds2df_client.produce(inputs = input_dataset)

    # SIMON client
    # try with no hyperparameter
    volumes = {} # d3m large primitive architecture dictionary of large files
    volumes['simon_models'] = '/home/simon_models'
    simon_client = simon(hyperparams={'overwrite':False}, volumes = volumes)

    # produce method
    result = simon_client.produce(inputs = df.value)
    print(result.value)

    # produce_metafeatures method
    features = simon_client.produce_metafeatures(inputs = df.value)
    print(features.value)
