import json
import os

import numpy as np
import pandas as pd
from Bio import AlignIO, Phylo, SeqIO
from Bio.Phylo.TreeConstruction import (DistanceCalculator,
                                        DistanceTreeConstructor)
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

import pyvolve
from buildNJMatrices import buildNJMatrices
from compareTrees import compareTrees
from dendropy.simulate import treesim
from ete3 import Tree
from generateTrainingData import *
from makePredictions import *
from trainModels import *


# SIMPLE EXAMPLE SIMULATORS
def exampleTreeGenerator(numberOfLeaves, numberOfTrees, outputDir, randomSeed=1):
    random.seed(randomSeed)
    for i in range(numberOfTrees):
        birth_rate = random.gauss(0.1, 0.01)
        death_rate = random.gauss(0.1, 0.01)
        tree = treesim.birth_death_tree(birth_rate=birth_rate, death_rate=death_rate, num_extant_tips=numberOfLeaves)
        fileIndexString = ('0000000'+str(i+1))[-7:]
        outputFile = (outputDir+'treeNum'+fileIndexString+'_'+str(numberOfLeaves)+'taxa_'+str(round(birth_rate,3))+'br_'+str(round(death_rate,3))+'dr'+'.nwk')
        tree.write(path=outputFile, schema="newick", suppress_rooting=True)

def exampleFastaGenerator(nwkFile, fastaOutputLocation, seqLength, rate=1):
    # Tree.
    treeName = nwkFile[nwkFile.rindex('/'):]
    treeName = treeName.split('.')[0]
    phylogony = pyvolve.read_tree(file=nwkFile)
    # Rates.
    mutationRates = {"AC":rate, "AG":rate, "AT":rate, "CG":rate, "CT":rate, "GT":rate}
    # Model.
    model = pyvolve.Model("nucleotide", {"mu": mutationRates})
    partition = pyvolve.Partition(models = model, size = seqLength)
    # Evolver.
    evolver = pyvolve.Evolver(partitions = [partition], tree = phylogony)
    evolver(seqfile = fastaOutputLocation, ratefile = None, infofile = None)


def phynn(runName, numberOfLeaves, numberOfSites, numberOfTrees, treeGenerator, evolver, NNParameters):
    ''' The main function for building and testing a model.

    Takes input parameters as well as:
        treeGenerator - A function that generates a random newick tree (can be defined however you want)
        evolver - A function that takes a newick trees and generates a fasta of the leaf nodes (ca be defined however you want)
            note: these two functions above should have a cleaner interface but the way they are set up (see the examples above) works for now

    Returns:
        A printout of how well it performed on a subset of your simulated data compared to nj-trees and "random" trees (one tree from the subset vs. another)
        All the intermediate files as well as the resulting files/models/resultmetrics get saved to the 'data' folder
        Optional: it could easily be configured to return a function that takes fasta and returns predicted newick
    '''
    print('\ngenerating trainng data------------------------')
    generateTrainingData(runName, numberOfLeaves, numberOfSites, numberOfTrees, treeGenerator, evolver)
    print('\ntraining models--------------------------------')
    trainModels(runName, NNParameters)
    print('\nmake predictions-------------------------------')
    makePredictions(runName)
    print('\nbuilding nj trees------------------------------')
    exampleFastaMatrices = np.load('data/np/'+runName+'_testTrueFastaMatrices.npy')
    buildNJMatrices(runName, exampleFastaMatrices, 'data/nj/')
    print('\ncomparing trees--------------------------------')
    compareTrees(runName)

# Example Invocation (can be called by running this file)
phynn(
        runName = 'ExampleRun',
        numberOfLeaves = 6,
        numberOfSites = 20,
        numberOfTrees = 1000,
        treeGenerator = exampleTreeGenerator,
        evolver = exampleFastaGenerator,
        NNParameters = {
            'numberFilters': 30,
            'kernalSize': 'onePerSite',
            'epocs': 30
            }
        )
