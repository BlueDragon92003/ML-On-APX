import uproot
from cluster_classification_dataset import ClusterClassificationDataset
from cluster import ClusterType
import time

tree = (uproot.open('../data/classification/double_electron.root'))["l1NtupleProducer/linkTree;1"]

tic = time.perf_counter()
ccd = ClusterClassificationDataset( { ( '../data/classification/double_electron.root', ClusterType.BACKGROUND ) } )
toc = time.perf_counter()
print(f"{toc - tic:0.4f}")
