import json

import numpy as np


def compareTrees(runName, verbose=True):
    '''Simple comparison of rmse difference between the tree matrices...
        this can be expanded to use another tree difference method,
        preferably one unrelated to how the NN model is trained to provide a more objective metric of performance '''

    # Load the data.
    # Topo
    trueTopo = np.load('data/np/'+runName+'_testTrueTopo.npy')
    shuffledTrueTopo = np.load('data/np/'+runName+'_testTrueTopo.npy')
    np.random.shuffle(shuffledTrueTopo)
    njTopo = np.load('data/nj/'+runName+'_NJtopologyMatrices.npy')
    nnTopo = np.load('data/np/'+runName+'_testPredictionsTopo.npy')
    # Dist
    trueDist = np.load('data/np/'+runName+'_testTrueDist.npy')
    shuffledTrueDist = np.load('data/np/'+runName+'_testTrueDist.npy')
    np.random.shuffle(shuffledTrueDist)
    njDist = np.load('data/nj/'+runName+'_NJdistanceMatrices.npy')
    nnDist = np.load('data/np/'+runName+'_testPredictionsDist.npy')

    #RMSE (root-mean-squared-error)
    def rmseNP(np1, np2):
        ''' Take two one-dimensional numpy arrays
            Return the root-mean-squared-error
            '''
        return np.sqrt(((np1 - np2) ** 2).mean())

    def averageRmseNP(np1, np2):
        ''' Take two two-dimensional numpy arrays
            Loop over the first dimension
            calculate the rmse between the elements
            return the average of all the elements'''
        numberOfElements = np1.shape[0]
        totalRMSE = 0
        for i in range(numberOfElements):
            totalRMSE+=rmseNP(np1[i], np2[i])
        averageRMSE = totalRMSE/numberOfElements
        return averageRMSE

    #MAE (mean-absolute-error)
    def maeNP(np1, np2):
        ''' Take two one dim np arrays
            Return the mean-absolute-error
        '''
        return np.sum(np1-np2)/len(np1)

    def averageMae(np1, np2):
        ''' Take two two-dim np arrays
            loop over the first dim
            calcualte the MAE between the elements
            return the average'''
        numberOfElements = np1.shape[0]
        totalMAE = 0
        for i in range(numberOfElements):
            totalMAE+=maeNP(np1[i]-np2[i])
        averageMAE = totalMAE/numberOfElements
        return averageMAE


    topologyComparison = {
            'nj-true': averageRmseNP(njTopo, trueTopo),
            'nn-true': averageRmseNP(nnTopo, trueTopo),
            'shuffledTrue-True': averageRmseNP(trueTopo, shuffledTrueTopo)
            }

    distanceComparison = {
            'nj-true': averageRmseNP(njDist, trueDist),
            'nn-true': averageRmseNP(nnDist, trueDist),
            'shuffledTrue-True': averageRmseNP(trueDist, shuffledTrueDist)
            }

    treeComparison = {
            'topologyComparison': topologyComparison,
            'distanceComparison': distanceComparison
            }

    with open(('data/results/'+runName+'_results.json'), 'w') as outfile:
        json.dump(treeComparison, outfile)


    if verbose:
        print('Topology Predictions:\n')
        print('nj vs true: ', round(averageRmseNP(njTopo, trueTopo), 2))
        print('nn vs true: ', round(averageRmseNP(nnTopo, trueTopo), 2))
        print('---compare to random---')
        print('true vs shuffled: ', round(averageRmseNP(trueTopo, shuffledTrueTopo), 4))

        print('\nBranch Length Predictions:\n')
        print('nj vs true: ', round(averageRmseNP(njDist, trueDist), 2), ' ...Something is wrong with the nj branch length predictions...')
        print('nn vs true: ', round(averageRmseNP(nnDist, trueDist), 2))
        print('---compare to random---')
        print('true vs shuffled: ', round(averageRmseNP(trueDist, shuffledTrueDist), 4))
