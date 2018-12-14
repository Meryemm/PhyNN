import os

import numpy as np
from Bio import AlignIO, Phylo, SeqIO
from Bio.Phylo.TreeConstruction import (DistanceCalculator,
                                        DistanceTreeConstructor)
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ete3 import Tree
from generateTrainingData import (flattenSymetricalMatrix, newickToMatrix,
                                  oneHotEncodeDNA, oneHotEncodeFasta)
from toFromMatrix import *


def matrixNPToFasta(fastaMatrixNP, outputFilePath):
    fastaMatrix = fastaMatrixNP.tolist()
    matrixToFasta(fastaMatrix, outputFilePath)

def fastaToNJTree(fastaFile, outputFile):
    aln = AlignIO.read(fastaFile, 'fasta')
    calculator = DistanceCalculator('identity')
    dm = calculator.get_distance(aln)
    constructor = DistanceTreeConstructor(calculator, 'nj')
    tree = constructor.build_tree(aln)
    Phylo.write(tree, outputFile, 'newick')


#------------------------------------------------------------------------------

def buildNJMatrices(runName, fastaMatrices, outputDirectory):
    ''' This is a bit hacky right now... idealy it would take the fasta files and build the nj trees directly from those...
        We are going back from the fasta matrices to easily keep the order of fasta files form the test dataset. can be changed later.

        This function will:
        Take a numpy array "fastaMatrices" whos first dimension is the number of fastas encoded.
        convert these onehotencoded fastas back into standard fastas and save the fastas to the output directory
        build NJ trees for each of these fastas
        convert the NJ trees to the matrix representations (topology and distance)
        concatenate all the matrix representations into two np arrays (topology and distance)
        save those two np arrays to files
        '''
    # Convert back to standard fasta.
    for i in range(fastaMatrices.shape[0]):
        indexString = ('0000'+str(i+1))[-4:]
        matrixNPToFasta(fastaMatrices[i], (outputDirectory+runName+'_'+indexString+'.fasta'))

    # Build NJ trees.
    directory = os.fsencode(outputDirectory)
    fileNameList = [os.fsdecode(file) for file in os.listdir(directory)]
    fileNameList.sort()
    for filename in fileNameList:
        if filename.endswith(".fasta"):
            fastaToNJTree((outputDirectory+filename), (outputDirectory+filename+'_NJ.nwk'))

    # Convert the NJ trees into the respective matrices (topo and distance).
    topologyMatrices = []
    directory = os.fsencode(outputDirectory)
    fileNameList = [os.fsdecode(file) for file in os.listdir(directory)]
    fileNameList.sort()
    for filename in fileNameList:
        if filename.endswith(".nwk"):

            topologyMatrix = newickToMatrix(outputDirectory+filename, type='topology')
            flattenedTopologyMatrix = flattenSymetricalMatrix(topologyMatrix)
            topologyMatrices.append(flattenedTopologyMatrix)
    # Convert the matrix of all the trees into a numpy array.
    topologyMatricesNP = np.array([np.array(x) for x in topologyMatrices])
    print('Number of trees in test set (i.e. topologyMatricesNP.shape[0]): ', topologyMatricesNP.shape[0])
    np.save(outputDirectory+runName+'_NJtopologyMatrices', topologyMatricesNP)

    distanceMatrices = []
    directory = os.fsencode(outputDirectory)
    fileNameList = [os.fsdecode(file) for file in os.listdir(directory)]
    fileNameList.sort()
    for filename in fileNameList:
        if filename.endswith(".nwk"):

            distanceMatrix = newickToMatrix(outputDirectory+filename, type='distace')
            flattenedDistanceMatrix = flattenSymetricalMatrix(distanceMatrix)
            distanceMatrices.append(flattenedDistanceMatrix)
    # Convert the matrix of all the trees into a numpy array.
    distanceMatricesNP = np.array([np.array(x) for x in distanceMatrices])
    np.save(outputDirectory+runName+'_NJdistanceMatrices', distanceMatricesNP)
