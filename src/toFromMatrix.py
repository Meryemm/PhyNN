import ast
import os
import random

import numpy as np
from Bio import AlignIO, Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ete3 import Tree


# ----- FASTA TO MATRIX -----
def oneHotEncodeDNA(sequence):
    # A function to convert a dna sequence (list ['A','A','C', etc.]) into a one hot encoding version.
    baseEncodings = {
        'A': [1,0,0,0],
        'a': [1,0,0,0],
        'C': [0,1,0,0],
        'c': [0,1,0,0],
        'G': [0,0,1,0],
        'g': [0,0,1,0],
        'T': [0,0,0,1],
        't': [0,0,0,1],
        '-': [0,0,0,0],
        '?': [0.25,0.25,0.25,0.25],
        '*': [0.25,0.25,0.25,0.25],
        'B': [0,0.33,0.33,0.33],
        'D': [0.33,0,0.33,0.33],
        'H': [0.33,0.33,0,0.33],
        'K': [0,0,0.5,0.5],
        'M': [0.5,0.5,0,0],
        'N': [0.25,0.25,0.25,0.25],
        'R': [0.5,0,0.5,0],
        'S': [0,0.5,0.5,0],
        'V': [0.33,0.33,0.33,0],
        'W': [0.5,0,0,0.5],
        'Y': [0,0.5,0,0.5]
                    }
    encodedSequence = []
    for base in sequence:
        encodedSequence.append(baseEncodings[base])
    return encodedSequence

def oneHotEncodeFasta(fastaFile):
    # A function to convert a fasta file into a one hot encoding version.
    # Encode the sequences.
    fastaData = SeqIO.parse(fastaFile, "fasta")
    encodedSequences = []
    ids = []
    for record in fastaData:
        sequence = list(record.seq)
        encodedSequence = oneHotEncodeDNA(sequence)
        encodedSequences.append(encodedSequence)
        ids.append(record.id)
    # Sort by ID.
    ids, encodedSequences = (list(x) for x in zip(*sorted(zip(ids, encodedSequences), key=lambda pair: pair[0])))
    return encodedSequences

# ----- MATRIX TO FASTA -----
def oneHotUnEncodeDNA(sequence):
    # A function to convert a one hot encoded dna sequence back to a (list ['A','A','C', etc.])
    baseEncodings = {
        'A': [1,0,0,0],
        #'a': [1,0,0,0],
        'C': [0,1,0,0],
        #'c': [0,1,0,0],
        'G': [0,0,1,0],
        #'g': [0,0,1,0],
        'T': [0,0,0,1],
        #'t': [0,0,0,1],
        '-': [0,0,0,0],
        '?': [0.25,0.25,0.25,0.25],
        '*': [0.25,0.25,0.25,0.25],
        'B': [0,0.33,0.33,0.33],
        'D': [0.33,0,0.33,0.33],
        'H': [0.33,0.33,0,0.33],
        'K': [0,0,0.5,0.5],
        'M': [0.5,0.5,0,0],
        'N': [0.25,0.25,0.25,0.25],
        'R': [0.5,0,0.5,0],
        'S': [0,0.5,0.5,0],
        'V': [0.33,0.33,0.33,0],
        'W': [0.5,0,0,0.5],
        'Y': [0,0.5,0,0.5]
                    }
    encodingKeys = list(baseEncodings.keys())
    encodingValues = list(baseEncodings.values())
    unEncodedSequence = []
    for encodedBase in sequence:
        base = encodingKeys[encodingValues.index(encodedBase)]
        unEncodedSequence.append(base)
    return unEncodedSequence

def matrixToFasta(fastaMatrix, outputFilePath):
    # A function to convert a fasta matrix into a fasta file
    # Unencode the sequences.
    unEncodedSequences = []
    for seq in fastaMatrix:
        unEncodedSeq = oneHotUnEncodeDNA(seq)
        unEncodedSequences.append(unEncodedSeq)
    # Write the fasta file.
    records = []
    idIndex = 1
    for unEncodedSeq in unEncodedSequences:
        unEncodedSeqString = ''.join(unEncodedSeq)
        idIndexString = ('0000'+str(idIndex))[-4:]
        record = SeqRecord(Seq(unEncodedSeqString), id=idIndexString)
        records.append(record)
        idIndex+=1
    SeqIO.write(records, outputFilePath, "fasta")


# ----- NEWICK TO MATIRCES -----
def getLeafNodeNames(tree):
    # return a list of leaf node names in alphabetical order
    leafNodeNames = []
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            leafNodeNames.append(node.name)
    leafNodeNames.sort()
    return leafNodeNames


def newickToMatrix(newickFile, type):
    '''Take a newick file return a matrix of distances between leaf nodes.
        the distance will either be number of internal nodes between leaves or
        total branch length between leaves (depending on what is set as `type`'''
    with open(newickFile,'r') as f:
        newickString = f.read()
    tree = Tree(newickString.replace(';',' ')+';', format=1)

    leafNodeNames = getLeafNodeNames(tree)

    distanceMatrix = []
    for leafNode in leafNodeNames:
        distancesFromNode = []
        for otherLeafNode in leafNodeNames:
            if leafNode == otherLeafNode:
                distance = 0
            else:
                if type == 'topology':
                    distance = int(tree.get_distance(leafNode, otherLeafNode, topology_only=True))
                else:
                    distance = tree.get_distance(leafNode, otherLeafNode)
            distancesFromNode.append(distance)
        distanceMatrix.append(distancesFromNode)
    return(distanceMatrix)

# ----- MATRICES TO NEWICK -----
#TODO: Matrix to newick
#TODO: "fuzzy" Matrix to newick
#Done: Topo Matrix to newickLike string

def matrixToDict(matrix, leafNames=False, rounding=False):
    '''Convert a symetrical matrix to a dictionary of dictonaries.'''
    if leafNames:
        leaves = leafNames
    else:
        leaves = [str(i) for i in range(len(matrix))]
    if rounding:
        matrix.round()
    dic = dict((key, value) for (key, value) in zip(leaves, matrix))
    for key in dic:
        dic[key] = dict((key, value) for (key, value) in zip(leaves, dic[key]))
    return dic

def createInternalNode(topologyDict, node1, node2):
    internalNodeName = "[" + node2 + "," + node1 + "]"
    topologyDict[internalNodeName] = {key: value-1 for (key, value) in topologyDict[node1].items()}
    del(topologyDict[node1])
    del(topologyDict[node2])
    for outerNode in topologyDict:
        del(topologyDict[outerNode][node1])
        del(topologyDict[outerNode][node2])
        if outerNode == internalNodeName:
            topologyDict[outerNode][internalNodeName] = 0
        else:
            topologyDict[outerNode][internalNodeName] = topologyDict[internalNodeName][outerNode]

    return topologyDict

def topologyMatrixToList(topologyMatrix):
    topologyDict = matrixToDict(topologyMatrix)

    # Basically three loops:
    #   1. loop until the list is complete / the whole tree is traversed
    #   2. loop through rows
    #   3. loop through the columns of each row

    # Loop 1
    unresolvedInternalNodes = True
    continueLooping = 0
    while unresolvedInternalNodes:


        newRowsToInsert = []

        # Loop 2
        # Go through each row; not using the actual dict because it will be modified mid loop.
        rowKeys = list(topologyDict.keys())
        for rowKey in rowKeys:
            if rowKey in list(topologyDict.keys()): # To handle looping over a dict that's being modified.

                # Loop 3
                colKeys = list(topologyDict[rowKey].keys())
                for colKey in colKeys:
                    if rowKey in list(topologyDict.keys()) and colKey in list(topologyDict[rowKey].keys()): # To handle looping over a dict that's being modified.

                        # Check if it is two nodes that share one internal node.
                        if topologyDict[rowKey][colKey] == 1:

                            # Add the internal node to the list
                            internalNodeName = "[" + rowKey + "," + colKey + "]"
                            internalNodeDict = {key: value-1 for (key, value) in topologyDict[rowKey].items()}
                            del(internalNodeDict[rowKey])
                            del(internalNodeDict[colKey])
                            newRowsToInsert.append({internalNodeName: internalNodeDict})

                            # Remove the two rows representing the two nodes that were merged into the internal node.
                            del(topologyDict[rowKey])
                            del(topologyDict[colKey])
                            # Remove the two columns representing the two nodes that were merged (do this for each remaining row)
                            for remainingRowKey in topologyDict:
                                del(topologyDict[remainingRowKey][rowKey])
                                del(topologyDict[remainingRowKey][colKey])

        # After each time looping throught all rows.
        # Add the new internal node(s) to the topologyMatrix (both as a new row and a new column to each row)
        for newRow in newRowsToInsert:
            [(newRowKey, newRowVal)] = newRow.items()
            topologyDict[newRowKey] = newRowVal

            for row in topologyDict:
                if row in list(newRowVal.keys()):
                    topologyDict[row][newRowKey] = newRowVal[row]
                else:
                    topologyDict[row][newRowKey] = 0

        # if there are three or less nodes left than just combine these nodes and you have the topology list.
        numberOfUnresolvedNodes = len(list(topologyDict.keys()))
        if numberOfUnresolvedNodes <= 3:
            topologyListString = '['
            for nodeName in list(topologyDict.keys()):
                topologyListString += nodeName + ','
            topologyListString = topologyListString[:-1] + ']'
            unresolvedInternalNodes = False

    # Convert the topologyListString '[[1,2],[3,4]]' into an actual list [[1,2],[3,4]]
    topologyList = ast.literal_eval(topologyListString)

    return topologyList


# ----- FLATTEN MATRIX -----
def flattenSymetricalMatrix(matrix):
    ''' Take a 2d symetrical matrix (list of lists).
        Return a flattened version (single list)
        for example:
           input [[0,2,3],
                  [2,0,4],
                  [3,4,0]]
           output [2,3,4]
                  '''
    flattenedMatrix = []
    for i in range(len(matrix)):
        if i == 0:
            pass
        else:
            for elm in matrix[i][0:i]:
                flattenedMatrix.append(elm)
    return flattenedMatrix


# ----- UNFLATTEN MATRIX -----
# TODO
