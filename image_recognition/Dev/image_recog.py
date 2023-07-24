
# Importing the required libraries for the project

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import SPSA
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer, transpile,execute
from qiskit.circuit.library import RealAmplitudes
from sklearn.metrics import accuracy_score,log_loss
from qiskit.circuit import ParameterVector


# Dev Note :- Each image has Intensity from 0 to 255

''' All steps are explained below
    1. Fetch the MNIST dataset
    2. Access the data and target  
        i) filter the data with 7 and 9 ( for initial development we are doing binary classification )
    3. Split the data into train and test data ( 80% for training and 20% for testing )
    4. pca for dimensionality reduction (x_scaled_data is the scaled data)
    5. batching the data ( we are not using batching for now)
    6. custom varational circuit is defined for now instead of QCNN (var_circuit)
        i)   it will be updated once we have the QCNN
    7. Input data is encoded into quantum state and then passed through the custom variational circuit(qcnn_model)
   
   Pending :- loss function is not minimized proply to get convergance and possible reasons are 
        i) we are not encoding the data properly into quantum state 
        ii) expectation value calcuation needs to be improved
        ii) cobyla optimizer coulen't find the minimum value
        iv) we need to check if we need to use classical neural network to extract labels from the circuit output
        
    8. loss function is defined (loss) as our objective is to minimize the loss ( Currently changes are in progess)
        i)   Function to predict the label ( 7 or 9 i.e, 0,1) from circuit output is 
        ii)  loss function is pending and will be updated once the above function is done
        iii) need to check if to use classical neural network to extract labels from the circuit output
    9. Optimizer is defined (optimizer) 
    10. Testing the model (Pending :-  Improve the code to test the model on test data)
    '''

# Fetch the MNIST dataset from openml
mnist = fetch_openml('mnist_784', parser='auto')

# Access the data and target
# x has all the pixel values of image and has Data shape of (70000, 784)  here 784 is 28*28
x = mnist.data
print("shape of x:",x.shape)

# y has all the labels of the images and has Target shape of (70000,)
y = mnist.target
print("shape of y:",y.shape)


# convert the given y data to integers so that we can filter the data
y = y.astype(int)

# Filtering only values with 7 or 9 as we are doing binary classification for now and we will extend it to multi-class classification
binary_filter = (y == 7) | (y == 9)

# Filtering the x and y data with the binary filter
x = x[binary_filter]
y = y[binary_filter]

# create a new y data with 0 and 1 values with 7 as 0 and 9 as 1 so that we can use it for binary classification
y = (y == 9).astype(int)


''' Here X_train has all the training pixel values of image (i.e, 80% ) and has Data shape of (56000, 784) here 784 is 28*28
    Here y_train has all the training labels of the images (i.e, 80% ) and has Target shape of (56000,)
    
    Here X_test has all the testing pixel values of images (i.e, 20% ) and has Data shape of (14000, 784) 
    Here y_test has all the testing labels of the images (i.e, 20% ) and has Target shape of (14000,)'''

# Here we have only two classes data out of above mentioned data    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x.shape, y.shape, x_train.shape, y_train.shape, x_test.shape, y_test.sha?pe)

# Reduce the size of the dataset to 200 training samples and 50 test samples for faster execution 
# will be removed once the code is ready for final execution
x_train = x_train[:200]
y_train = y_train[:200]
x_test  = x_test[:50]
y_test  = y_test[:50]



# Number of qubits for the quantum circuit
num_qubits = 14

# After initial development  below code will be move to qubit loop
# -------------- PCA Compression -----------------
# Number of principal components to keep as of now it is 14 
pca = PCA(n_components = num_qubits)
# pca = PCA( ) if pca_check == True else PCA(n_components = num_qubits)


# Apply PCA on the data to reduce the dimensions and it is in the form of 'numpy.ndarray'
x_train_pca = pca.fit_transform(x_train)
x_test_pca =  pca.fit_transform(x_test)

# To visualize if prinicipal components are enough and to decide the number of principal components to keep 
pca_check = False 
if pca_check == True:
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()


# Create an instance of MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# 2PIE

# Apply min-max scaling to the data to bring the values between 0 and 1
x_scaled_test =  scaler.fit_transform(x_test_pca)
x_scaled_train = scaler.fit_transform(x_train_pca)

# ------------Dev Note Currently we are not using batching-----------------

# Number of samples per batch
samples_per_batch = 10

# Calculate the total number of batches needed
num_batches = len(x_scaled_train) // samples_per_batch

# Batch the x_scaled_data array
x_scaled_test_batches = np.array_split(x_scaled_test, num_batches)

# print(x_scaled_test_batches[0].shape)

# for batch in x_scaled_test_batches[:3]:
#     print("batch",batch)

#----------- Dev Note Currently we are not using batching------------------

# Data frame here to just to visualize the data and Not needed once model is defined
pca_vis = False
if pca_vis == True:
    x_vis = pd.DataFrame(x_scaled_train)
    stats = x_vis.describe()
    print(stats)


# Variational circuit  used in below model which has parameters optimized during training
# Dev Note will be replaced with QCNN 
def var_circ(num_qubits=num_qubits):
    
    # reps is Number of times ry and cx gates are repeated
    reps = 3
    qc = QuantumCircuit(num_qubits)
    parameter_vector = ParameterVector("t", length=num_qubits*reps)
    # print("parameter_vector",parameter_vector)
    counter = 0
    for rep in range(reps):
      for i in range(num_qubits):
          theta = parameter_vector[counter]
          qc.ry(theta, i)
          counter += 1
      for j in range(0, num_qubits - 1, 2):
          qc.cx(j, j + 1)
    # print("counter",counter)
    return qc


# var_circ for reference
#      ┌──────────┐ ░            ░ ┌──────────┐ ░            
# q_0: ┤ RY(θ[0]) ├─░───■────────░─┤ RY(θ[3]) ├─░───■────────
#      ├──────────┤ ░ ┌─┴─┐      ░ ├──────────┤ ░ ┌─┴─┐      
# q_1: ┤ RY(θ[1]) ├─░─┤ X ├──■───░─┤ RY(θ[4]) ├─░─┤ X ├──■───
#      ├──────────┤ ░ └───┘┌─┴─┐ ░ ├──────────┤ ░ └───┘┌─┴─┐ 
# q_2: ┤ RY(θ[2]) ├─░──────┤ X ├─░─┤ RY(θ[5]) ├─░──────┤ X ├─
#      └──────────┘ ░      └───┘ ░ └──────────┘ ░      └───┘ 




# model to be used for training which has input data encoded and variational circuit is appended to it
def qcnn_model(theta, x):
    qc = QuantumCircuit(num_qubits, num_qubits//2)
    
    # Encode the pixel data into the quantum circuit here  x is the input data which is list of 14 values
    
    
    # feature mapping 
    for j in range(num_qubits):
        qc.ry(x[j], j )
        
    # Append the variational circuit ( Ansatz ) to the quantum circuit
    qcnn_circ = var_circ(num_qubits)
    qcnn_circ.assign_parameters(theta, inplace=True)
    qc.compose(qcnn_circ, inplace=True)
    # qc.measure_all()  # Measure all qubits will be changed to measure only 7 qubits if needed
    # Measure only the first 7 qubits
    # Add a classical register with 7 bits to store the results of the measurements
    for i in range(num_qubits//2):
        # print("i",i, "num_qubits//2",num_qubits//2)
        qc.measure(i, i)
    return qc


# Define the quantum instance to be used for training
backend = Aer.get_backend('qasm_simulator')

# threshold to be used for prediction of label from the circuit output
global threshold 

threshold = 0.5

# Number of shots to run the program (experiment)
num_shots = 100

# function to calculate the expectation value ( pending :- Analysis in progress to improve or replace below function )
def expectation_values(result):
    expectation = 0

    for outcome, count in result.items():
        bitstring = outcome#[::-1]  # Reverse the bitstring
        decimal_value = int(bitstring, 2)  # Convert the bitstring to decimal
        probability = count / num_shots  # Compute the probability of each bitstring
        expectation += decimal_value * probability / 2 ** (num_qubits/2)  # Compute the expectation value
    if expectation > threshold:
        prediction_label = 1
    else:
        prediction_label = 0
    return prediction_label
prediction_label = []


# function to calculate the loss function will be update after finding the suitable loss function
 # Tried applying log loss ( cross entropy) but it is not minimizing the loss
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

# function to calculate the loss function
def loss_function(theta):
    # predicted_label = []
    total_loss = 0
    epsilon = 1e-15  # small value to avoid log(0) errors
    prediction_label = []
    for data_point, label in zip(x_scaled_train, y_train):
        # Create the quantum circuit for the data point
        qc = qcnn_model(theta, data_point)
        # Simulate the quantum circuit and get the result 
        job = backend.run(qc, shots=num_shots)
        result = job.result().get_counts(qc)
        predicted_label = expectation_values(result)
        prediction_label.append(predicted_label)
        
    # Cross entropy loss
    loss = log_loss(y_train, prediction_label)
    print("cross entropy loss:", loss)
    loss = square_loss(y_train, prediction_label)
    print("loss:", loss)
    return loss

# Initialize  epochs
num_epochs = 10
reps = 3

# Initialize the weights for the QNN model
np.random.seed(0)
weights = np.random.rand(num_qubits * reps )
print(len(weights))
# Will increase the number of epochs once the code is fine tuned to get convergance 
for epoch in range(num_epochs):
    # Minimize the loss using SPSA optimizer
    theta = minimize(loss_function, x0 = weights, method="COBYLA", tol=0.001, options={'maxiter': 10, 'disp': False} )
    loss = theta.fun
    print(f"Epoch {epoch+1}/{num_epochs}, loss = {loss:.4f}")
    
# To find threshold
# print("max_values",max_values)
# max = max(max_values)
# average_value = sum(max_values) / len(max_values)
# print("average",average_value)
# last step is to test the model on test data

# Below code is to test the model on test data after training and getting theta values
# Perform inference with the trained QNN
predictions = []

# Loop over the test data and use theta obtained from training to get the predicted label
for data_point in x_scaled_test:
    qc = qcnn_model(theta.x, data_point)
    job = backend.run(qc, shots=100)
    result = job.result().get_counts(qc)
    prediction_label = expectation_values(result)
    predictions.append(prediction_label)

# Evaluate the QCNN accuracy on the test set once the model is trained and tested
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# print(("y_test",y_test), "predictions",predictions)