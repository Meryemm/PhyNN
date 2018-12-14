import os
import random

import numpy as np
from Bio import SeqIO

import pyvolve
from ete3 import Tree
from toFromMatrix import *

# ----- THE MAIN FUNCTION -----

def generateTrainingData(runName, numberOfLeaves, numberOfSites, numberOfTrees, treeGenerator, evolver):
    ''' It would be nice to break this out into a snakemake file
    but for expediency, It's just chaining some functions together for now
    '''

    # ----- SIMULATION -----
    # Simulate Trees.
    treeDir = ('data/trees/'+runName)
    os.mkdir(treeDir)
    treeGenerator(numberOfLeaves, numberOfTrees, treeDir+'/')

    # Simulate evolution for each tree saving the leaf nodes to fasta file.
    fastaDir = ('data/fasta/'+runName)
    os.mkdir(fastaDir)
    directory = os.fsencode(treeDir+'/')
    fileNameList = [os.fsdecode(file) for file in os.listdir(directory)]
    fileNameList.sort()
    for treeFileName in fileNameList:
        fastaFilePath = fastaDir+'/'+treeFileName+'.fasta'
        evolver((treeDir+'/'+treeFileName), fastaFilePath, seqLength=numberOfSites, rate=1)

    # ----- CONVERT TO MATRICES -----
    # Convert the trees to topology matrices.
    topologyMatrices = []
    directory = os.fsencode(treeDir+'/')
    fileNameList = [os.fsdecode(file) for file in os.listdir(directory)]
    fileNameList.sort()
    for filename in fileNameList:
        topologyMatrix = newickToMatrix(treeDir+'/'+filename, type='topology')
        flattenedTopologyMatrix = flattenSymetricalMatrix(topologyMatrix)
        topologyMatrices.append(flattenedTopologyMatrix)
    # Convert the matrix of all the trees into a numpy array.
    topologyMatricesNP = np.array([np.array(x) for x in topologyMatrices])
    print('topologyMatricesNP shape: ', topologyMatricesNP.shape)

    # Convert the trees to distance matrices.
    distanceMatrices = []
    directory = os.fsencode(treeDir+'/')
    fileNameList = [os.fsdecode(file) for file in os.listdir(directory)]
    fileNameList.sort()
    for filename in fileNameList:
        distanceMatrix = newickToMatrix(treeDir+'/'+filename, type='distance')
        flattenedDistanceMatrix = flattenSymetricalMatrix(distanceMatrix)
        distanceMatrices.append(flattenedDistanceMatrix)
    # Convert the matrix of all the trees into a numpy array.
    distanceMatricesNP = np.array([np.array(x) for x in distanceMatrices])
    print('distanceMatricesNP shape: ', distanceMatricesNP.shape)

    # Convert the simulated fasta files to matrices.
    fastaMatrices = []
    directory = os.fsencode(fastaDir+'/')
    fileNameList = [os.fsdecode(file) for file in os.listdir(directory)]
    fileNameList.sort()
    for filename in fileNameList:
        if filename.endswith(".fasta"):
            fastaMatrices.append(oneHotEncodeFasta(fastaDir+'/'+filename))
    # Covert the matrix of all the fasta files into a numpy array.
    fastaMatricesNP = np.array([np.array(x) for x in fastaMatrices])
    print('fastaMatricesNP shape: ', fastaMatricesNP.shape)

    # Save the numpy arrays to files.
    np.save('data/np/'+runName+'_fastaMatrices', fastaMatricesNP)
    np.save('data/np/'+runName+'_topologyMatrices', topologyMatricesNP)
    np.save('data/np/'+runName+'_distanceMatrices', distanceMatricesNP)
