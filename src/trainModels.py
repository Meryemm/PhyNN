import numpy as np
import pandas as pd
from IPython.display import SVG
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import train_test_split

np.random.seed(0)
random_seed = 0


def trainModels(runName, NNParameters, verbose=2):
    # Train two very simple keras models; one for topology, one for branch-length distances.

    # Set NN Parameters
    numberFilters = NNParameters['numberFilters']
    kernalSize = NNParameters['kernalSize']
    epocs = NNParameters['epocs']

    # Load the data
    fastaMatrix = np.load('data/np/'+runName+'_fastaMatrices.npy')
    topologyMatrix = np.load('data/np/'+runName+'_topologyMatrices.npy')
    distanceMatrix = np.load('data/np/'+runName+'_distanceMatrices.npy')

    if kernalSize == 'onePerSite':
        # Set the kernalSize to (number of sequences, 1)
        kernalSize = (fastaMatrix.shape[1], 1)

    # Test train split Topology.
    x_train, x_validation, y_train_topology, y_validation_topology = train_test_split(fastaMatrix, topologyMatrix, test_size = 0.1, random_state=random_seed)
    # Hold back some data completely
    x_train, x_test, y_train_topology, y_test_topology = train_test_split(x_train, y_train_topology, test_size = 0.02, random_state=random_seed)

    # Test train split Distance.
    x_train, x_validation, y_train_distance, y_validation_distance = train_test_split(fastaMatrix, distanceMatrix, test_size = 0.1, random_state=random_seed)
    # Hold back some data completely
    x_train, x_test, y_train_distance, y_test_distance = train_test_split(x_train, y_train_distance, test_size = 0.02, random_state=random_seed)

    # Define the models (very simple architecture for now: one hidden layer)
    inputShape = (x_train.shape[1:4])
    modelTopo = Sequential()
    modelTopo.add(Conv2D(filters = numberFilters, kernel_size = kernalSize, padding = 'valid', activation ='relu', input_shape=inputShape))
    modelTopo.add(Flatten())
    modelTopo.add(Dense(y_train_topology.shape[1], activation = "relu"))

    modelDist = Sequential()
    modelDist.add(Conv2D(filters = numberFilters, kernel_size = kernalSize,padding = 'valid', activation ='relu', input_shape = (x_train.shape[1:4]), data_format="channels_last"))
    modelDist.add(Flatten())
    modelDist.add(Dense(y_train_topology.shape[1], activation = "relu"))

    # Print out some summary info
    print('\n---Training Data Summary---\n')
    print('Number of Trees: ', x_train.shape[0])
    print('Number of leaf nodes per tree: ', x_train.shape[1])
    print('Number of sites per seq: ', x_train.shape[2])
    print('\n---Keras Model Parameter Summary---\n')
    print('   modelTopo and modelDist are both the same')
    print('Number of parameters in the model: ', modelTopo.count_params())
    print(modelTopo.summary())

    # Optimizer.
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile.
    modelTopo.compile(optimizer = optimizer , loss = "mean_squared_error", metrics=["mean_squared_error"])
    modelDist.compile(optimizer = optimizer , loss = "mean_squared_error", metrics=["mean_squared_error"])

    batch_size = 32

    history = modelTopo.fit(x_train, y_train_topology, batch_size=batch_size, epochs=epocs, validation_data = (x_test, y_test_topology), verbose = verbose, callbacks=[TensorBoard(log_dir='tf_logs')])
    history = modelDist.fit(x_train, y_train_distance, batch_size=batch_size, epochs=epocs, validation_data = (x_test, y_test_distance), verbose = verbose, callbacks=[TensorBoard(log_dir='tf_logs')])

    # Serialize models to file
    # The model architecture is saved as json
    modelJsonTopo = modelTopo.to_json()
    with open("data/keras/"+runName+"_modelTopo.json", "w") as json_file:
        json_file.write(modelJsonTopo)
    modelJsonDist = modelDist.to_json()
    with open("data/keras/"+runName+"_modelDist.json", "w") as json_file:
        json_file.write(modelJsonDist)
    # The model weights are saved as h5py
    modelTopo.save_weights("data/keras/"+runName+"_modelTopo.h5")
    modelDist.save_weights("data/keras/"+runName+"_modelDist.h5")

    # Save the true test fasta and topo/dist Matrices.
    np.save('data/np/'+runName+'_testTrueFastaMatrices', x_test)
    np.save('data/np/'+runName+'_testTrueTopo', y_test_topology)
    np.save('data/np/'+runName+'_testTrueDist', y_test_distance)
