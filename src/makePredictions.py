import numpy as np
from keras.models import model_from_json


def makePredictions(runName):
    # Load the true test fasta and topo/dist Matrices.
    testFastaMatrices = np.load('data/np/'+runName+'_testTrueFastaMatrices.npy')
    testTrueTopoMatrices = np.load('data/np/'+runName+'_testTrueTopo.npy')
    testTrueDistMatrices = np.load('data/np/'+runName+'_testTrueDist.npy')

    # Load the trained Keras models

    # load json topo model
    topo_json_file = open('data/keras/'+runName+'_modelTopo.json', 'r')
    loaded_topo_model_json = topo_json_file.read()
    topo_json_file.close()
    modelTopo = model_from_json(loaded_topo_model_json)
    # load weights into the topo model
    modelTopo.load_weights('data/keras/'+runName+'_modelTopo.h5')

    # load json dist model
    dist_json_file = open('data/keras/'+runName+'_modelDist.json', 'r')
    loaded_dist_model_json = dist_json_file.read()
    dist_json_file.close()
    modelDist = model_from_json(loaded_dist_model_json)
    # load weights into the dist model
    modelDist.load_weights('data/keras/'+runName+'_modelDist.h5')

    # Make and save predictions on the test data set.
    distPredictions = []
    topoPredictions = []
    for x in testFastaMatrices:
        expandedX = np.expand_dims(x, axis=0)
        topoPrediction = modelTopo.predict(expandedX)
        topoPredictions.append(topoPrediction)
        distPrediction = modelDist.predict(expandedX)
        distPredictions.append(distPrediction)
    topoPredictionsNP = np.concatenate(tuple(topoPredictions))
    distPredictionsNP = np.concatenate(tuple(distPredictions))
    # Save the predictions.
    np.save('./data/np/'+runName+'_testPredictionsTopo', topoPredictionsNP)
    np.save('./data/np/'+runName+'_testPredictionsDist', distPredictionsNP)
