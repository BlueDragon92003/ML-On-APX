import os
import pickle
from typing import Set, Tuple, Iterable, Iterator
from unittest.mock import patch
import pathlib
import unittest

from pyfakefs import fake_filesystem as fakefs
from pyfakefs.fake_filesystem_unittest import TestCaseMixin
import numpy as np

from cluster_classification.classification_logger import CleverLogger
from cluster_classification.data import load_data, DatasourceType
from cluster_classification.dataset_subset import DatasetSubset
from cluster_classification.signal_type import SignalType

logger = CleverLogger(__name__)

def logging_wrapper_generator(generator: Iterable):
    for next in generator:
        logger.log_debug(f'Logging from wrapped {generator}:\t{next}')
        yield next

def mock_cluster_generator(components: Iterable) -> Iterator[int]:
    return map(lambda a: sum([ byte for byte in str(a[0]).encode() ]),
        components
        )

class Mock_ClusterClassificationDataset():
    def __init__(self, components: Set[Tuple[str, SignalType]]):
        self.__data = np.fromiter(
            logging_wrapper_generator(mock_cluster_generator(components)),
            (int,2)
        )
        self.msg = "Oh, wow! Look at all of that testing data!"
    def __getitem__(self, key):
        return self.__data[key]
    def __repr__(self):
        return f"MockCCD: {self.__data}"

@patch( 'cluster_classification.data.ClusterClassificationDataset',
    new=Mock_ClusterClassificationDataset,
    )
class TestLoadData(unittest.TestCase, TestCaseMixin):
    """Tests for the `load_data` method in `data.py`.

    Extends: `unittest.TestCase`

    Public methods:
    - setUpClass @classmethod
    - test_load_data__pickle_exists
    - test_load_data__create_pickle
    - test_load_data__old_pickle_data
    - test_load_data__old_pickle_format

    """
    ffs: fakefs.FakeFilesystem

    def __init__(self, methodName='runTest'):
        super(TestLoadData, self).__init__(methodName=methodName)

    @classmethod
    def setUpClass(cls):
        logger.log_info('Setting up fake filesystem...')
        cls.setUpClassPyfakefs()

        cls.ffs = cls.fake_fs()
        cls.paths = {}
        cls.dirs = {}
        cls.datasets = {}
        cls.files = {}

        #                                            tests  clsify
        cls.paths['module'] = pathlib.Path(__file__).parent.parent
        cls.paths['project'] = cls.paths['module'].parent
        cls.paths['data'] = cls.paths['project'] / 'data'
        cls.paths['root'] = cls.paths['data'] / 'classification'
        cls.paths['pickle'] = cls.paths['data'] / 'pickled' / 'classification'

        cls.dirs['project'] = cls.ffs.create_dir(cls.paths['project'])
        cls.dirs['module'] = cls.ffs.create_dir(cls.paths['module'])
        cls.dirs['data'] = cls.ffs.create_dir(cls.paths['data'])
        cls.dirs['root'] = cls.ffs.create_dir(cls.paths['root'])
        cls.dirs['pickle'] = cls.ffs.create_dir(cls.paths['pickle'])

        cls.datasets['data_1'] = DatasetSubset(0b0001, 'data_1.root', SignalType.BACKGROUND)
        cls.datasets['data_2'] = DatasetSubset(0b0010, 'data_2.root', SignalType.BACKGROUND)
        cls.datasets['data_3'] = DatasetSubset(0b0100, 'data_3.root', SignalType.BACKGROUND)
        cls.datasets['data_n'] = DatasetSubset(0b1000, 'data_n.root', SignalType.BACKGROUND)
        
        # Age 0
        cls.files['data_1'] = cls.ffs.create_file(
            cls.paths['root'] / 'data_1.root'
        )
        cls.files['data_2'] = cls.ffs.create_file(
            cls.paths['root'] / 'data_2.root'
        )
        cls.files['data_3'] = cls.ffs.create_file(
            cls.paths['root'] / 'data_3.root',
        )
        
        # Age 1
        cls.files['old_fmt_pckl'] = cls.ffs.create_file(
            cls.paths['pickle'] / '0000' / '0005.pckl',
        )
        #   Increment age
        cls.files['data_1'].set_contents('')
        cls.files['data_2'].set_contents('')
        cls.files['data_3'].set_contents('')
        
        # Age 2
        cls.files['ccd'] = cls.ffs.create_file(cls.paths['module'] / 'cluster_classification_dataset.py')
        #   Increment age
        cls.files['data_1'].set_contents('')
        cls.files['data_2'].set_contents('')
        cls.files['data_3'].set_contents('')
        cls.files['old_fmt_pckl'].set_contents('')
        
        # Age 3
        cls.files['old_data_pckl'] = cls.ffs.create_file(
            cls.paths['pickle'] / '0000' / '0009.pckl'
        )
        cls.files['valid_pckl'] = cls.ffs.create_file(
            cls.paths['pickle'] / '0000' / '0003.pckl'
        )
        #   Increment age
        cls.files['data_1'].set_contents('')
        cls.files['data_2'].set_contents('')
        cls.files['data_3'].set_contents('')
        cls.files['old_fmt_pckl'].set_contents('')
        cls.files['ccd'].set_contents('')
        
        # Age 4
        cls.files['data_n'] = cls.ffs.create_file(
            cls.paths['root'] / 'data_n.root',
        )
        #   Increment age & set contents
        cls.files['data_1'].set_contents('')
        cls.files['data_2'].set_contents('')
        cls.files['data_3'].set_contents('')
        cls.files['old_fmt_pckl'].set_contents(
            pickle.dumps(Mock_ClusterClassificationDataset(
                set([
                    ( str(cls.paths['root'] / 'data_1.root'), SignalType.BACKGROUND ),
                    ( str(cls.paths['root'] / 'data_3.root'), SignalType.BACKGROUND )
                ])
            ))
        )
        cls.files['ccd'].set_contents('')
        cls.files['old_data_pckl'].set_contents(
            pickle.dumps(Mock_ClusterClassificationDataset(
                set([
                    ( str(cls.paths['root'] / 'data_1.root'), SignalType.BACKGROUND ),
                    ( str(cls.paths['root'] / 'data_n.root'), SignalType.BACKGROUND )
                ])
            ))
        )
        cls.files['valid_pckl'].set_contents(
            pickle.dumps(Mock_ClusterClassificationDataset(
                set([
                    ( str(cls.paths['root'] / 'data_1.root'), SignalType.BACKGROUND ),
                    ( str(cls.paths['root'] / 'data_2.root'), SignalType.BACKGROUND )
                ])
            ))
        )

        logger.log_info('Set up fake filesystem.')
    # ========================================================================
    # with pickle
    # ------------------------------------------------------------------------
    def test_load_data__pickle_exists(self):
        """Test that the system reads a present, in-date pickle"""
        old_time = os.path.getmtime( self.paths['pickle'] / '0000' / '0003.pckl' )
        out = load_data(
            DatasourceType.TESTING,
            self.datasets['data_1'] | self.datasets['data_2']
        )
        expected = np.fromiter( mock_cluster_generator([
                (str(self.paths['root'] / 'data_1.root'), SignalType.BACKGROUND),
                (str(self.paths['root'] / 'data_2.root'), SignalType.BACKGROUND),
            ]), (int,2) )
        new_time = os.path.getmtime( self.paths['pickle'] / '0000' / '0003.pckl' )
        # that the pickle has the right data
        self.assertIn( out[0], expected )
        self.assertIn( out[1], expected )
        # that the pickle wasn't recreated
        self.assertEqual( new_time , old_time )

    # ========================================================================
    # without pickle
    # ------------------------------------------------------------------------
    def test_load_data__create_pickle(self):
        """Test that the system creates a new pickle when needed"""
        out = load_data(
            DatasourceType.TESTING,
            self.datasets['data_2'] | self.datasets['data_n']
        )
        expected = np.fromiter( mock_cluster_generator([
                (str(self.paths['root'] / 'data_2.root'), SignalType.BACKGROUND),
                (str(self.paths['root'] / 'data_n.root'), SignalType.BACKGROUND),
            ]), (int,2) )
        # that the pickle has the right data
        self.assertIn( out[0], expected )
        self.assertIn( out[1], expected )
        # that the pickle was created
        self.assertTrue( os.path.isfile( self.paths['pickle'] / '0000' / '000a.pckl' ) )

    # ========================================================================
    # with outdated pickle
    # ------------------------------------------------------------------------
    def test_load_data__old_pickle_data(self):
        """Test that the system updates a pickle when the root dataset is changed"""
        old_time = os.path.getmtime( self.paths['pickle'] / '0000' / '0009.pckl' )
        out = load_data(
            DatasourceType.TESTING,
            self.datasets['data_1'] | self.datasets['data_n']
        )
        expected = np.fromiter( mock_cluster_generator([
                (str(self.paths['root'] / 'data_1.root'), SignalType.BACKGROUND),
                (str(self.paths['root'] / 'data_n.root'), SignalType.BACKGROUND),
            ]), (int,2) )
        new_time = os.path.getmtime( self.paths['pickle'] / '0000' / '0009.pckl' )
        # that the pickle has the right data
        self.assertIn( out[0], expected )
        self.assertIn( out[1], expected )
        # that the pickle was recreated
        self.assertGreater( new_time , old_time )

    # ========================================================================
    # with old pickle
    # ------------------------------------------------------------------------
    def test_load_data__old_pickle_format(self):
        """Test that the system updates a pickle when the underlying representation is updated"""
        old_time = os.path.getmtime( self.paths['pickle'] / '0000' / '0005.pckl' )
        out = load_data(
            DatasourceType.TESTING,
            self.datasets['data_1'] | self.datasets['data_3']
        )
        expected = np.fromiter( mock_cluster_generator([
                (str(self.paths['root'] / 'data_1.root'), SignalType.BACKGROUND),
                (str(self.paths['root'] / 'data_3.root'), SignalType.BACKGROUND),
            ]), (int,2) )
        new_time = os.path.getmtime( self.paths['pickle'] / '0000' / '0005.pckl' )
        # that the pickle has the right data
        self.assertIn( out[0], expected )
        self.assertIn( out[1], expected )
        # That the pickle was recreated
        self.assertGreater( new_time , old_time )

