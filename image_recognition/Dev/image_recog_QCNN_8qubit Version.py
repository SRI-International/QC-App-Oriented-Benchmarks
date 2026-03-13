
# Importing the required libraries for the project

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer
from sklearn.metrics import accuracy_score,log_loss
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


# Initialize the weights for the QNN model
np.random.seed(1)

global expectation_calc_method

# change the below variable to True if you want to use expectation_calc.py file for expectation value calculation
expectation_calc_method = True

from qiskit.primitives import Estimator

estimator = Estimator()

def exp_cal(state):
    
    op = SparsePauliOp("IIIIZIII")

    expectation_value = estimator.run(state, op, shots = 1000).result().values

# for shot-based simulation:
    # expectation_value = estimator.run(state, op, shots=100).result().values

    return expectation_value

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
   
   8):- loss function is not minimized proply to get convergance and possible reasons are 
        i) we are not encoding the data properly into quantum state 
        ii) expectation value calcuation needs to be improved
        ii) cobyla optimizer coulen't find the minimum value
        iv) we need to check if we need to use classical neural network to extract labels from the circuit output
        
    8. loss function is defined (loss) as our objective is to minimize the loss ( Currently changes are in progess)
        i)   Function to predict the label ( 0,1) from circuit output is 
        ii)  loss function is pending and will be updated once the above function is done
        iii) need to check if to use classical neural network to extract labels from the circuit output
    9. Optimizer is defined (optimizer) 
    
    10. Testing the model (Pending :-  Improve the code to test the model on test data)
    '''
    # Pending
    #   To Find common expectatation value code & other additional functionality for 8 qubits
    

global num_qubits

# Number of qubits for the quantum circuit
num_qubits = 8

# Fetch the MNIST dataset from openml
mnist = fetch_openml('mnist_784')

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
binary_filter = (y == 0) | (y == 1)

# Filtering the x and y data with the binary filter
x = x[binary_filter]
y = y[binary_filter]

# create a new y data with 0 and 1 values with 7 as 0 and 9 as 1 so that we can use it for binary classification
# y = (y == 9).astype(int)


''' Here X_train has all the training pixel values of image (i.e, 80% ) and has Data shape of (56000, 784) here 784 is 28*28
    Here y_train has all the training labels of the images (i.e, 80% ) and has Target shape of (56000,)
    
    Here X_test has all the testing pixel values of images (i.e, 20% ) and has Data shape of (14000, 784) 
    Here y_test has all the testing labels of the images (i.e, 20% ) and has Target shape of (14000,)'''

# Here we have only two classes data out of above mentioned data    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 100, train_size = 500, random_state=42)
# print(x.shape, y.shape, x_train.shape, y_train.shape, x_test.shape, y_test.sha?pe)

# Reduce the size of the dataset to 200 training samples and 50 test samples for faster execution 
# will be removed once the code is ready for final execution
# x_train = x_train[:200]
# y_train = y_train[:200]
# x_test  = x_test[:50]
# y_test  = y_test[:50]

# After initial development  below code will be move to qubit loop
# -------------- PCA Compression -----------------
# Number of principal components to keep as of now it is 14 
# pca = PCA(n_components = num_qubits)
# # pca = PCA( ) if pca_check == True else PCA(n_components = num_qubits)


# # Apply PCA on the data to reduce the dimensions and it is in the form of 'numpy.ndarray'
# x_train_pca = pca.fit_transform(x_train)
# x_test_pca =  pca.transform(x_test)

print(len(x_train), len(y_train) , len(x_test), len(y_test))
# Create an instance of MinMaxScaler
# scaler = StandardScaler( ).fit(x)
# x_scaled_train =  scaler.transform(x_train)
# x_scaled_test =  scaler.transform(x_test)

# Step 1: Apply PCA on the entire dataset
pca = PCA(n_components = num_qubits).fit(x)
x_pca = pca.transform(x)
x_train_pca  = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# Step 2: Apply MinMax scaling to the PCA-transformed data
scaler = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit(x_pca)
x_final_train =  scaler.transform(x_train_pca)
x_final_test  =  scaler.transform(x_test_pca)
# # Create an instance of MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
# x_scaled_train = scaler.fit_transform(x_train)
# x_scaled_test =  scaler.transform(x_test)




# Define the convolutional circuits
def conv_circ_1(thetas, first, second):
    # Your implementation for conv_circ_1 function here
    # print(thetas)
    conv_circ = QuantumCircuit(8)
    conv_circ.rx(thetas[0], first)
    conv_circ.rx(thetas[1], second)
    conv_circ.rz(thetas[2], first)
    conv_circ.rz(thetas[3], second)

    conv_circ.ry(thetas[10], first)
    conv_circ.ry(thetas[11], second)
    conv_circ.rx(thetas[12], first)
    conv_circ.rx(thetas[13], second)
    
    conv_circ.crx(thetas[4], second, first)  
    conv_circ.crx(thetas[5], first, second)
    conv_circ.rx(thetas[6], first)
    conv_circ.rx(thetas[7], second)
    conv_circ.rz(thetas[8], first)
    conv_circ.rz(thetas[9], second)
    
    conv_circ.ry(thetas[14], first)
    conv_circ.ry(thetas[15], second)
    conv_circ.rx(thetas[16], first)
    conv_circ.rx(thetas[17], second)
    # print(conv_circ)
    return conv_circ

# Define the pooling circuits
def pool_circ_1(thetas, first, second):
    # Your implementation for pool_circ_1 function here
    pool_circ = QuantumCircuit(8)
    pool_circ.crz(thetas[0], first, second)
    # pool_circ.x(second)
    pool_circ.crx(thetas[1], first, second)
    return pool_circ

def pool_circ_2(first, second):
    pool_circ = QuantumCircuit(8)
    pool_circ.crz(first, second)
    return pool_circ

# Quantum Circuits for Convolutional layers
def conv_layer_1(qc, thetas):
    qc = qc.compose(conv_circ_1(thetas, 0, 1))
    qc = qc.compose(conv_circ_1(thetas, 2, 3))
    qc = qc.compose(conv_circ_1(thetas, 4, 5))
    qc = qc.compose(conv_circ_1(thetas, 6, 7))
    return qc

def conv_layer_2(qc, thetas):
    qc = qc.compose(conv_circ_1(thetas, 0, 2))
    qc = qc.compose(conv_circ_1(thetas, 4, 6))
    return qc

def conv_layer_3(qc, thetas):
    qc = qc.compose(conv_circ_1(thetas, 0, 4))
    return qc

# Quantum Circuits for Pooling layers
def pooling_layer_1(qc, thetas):
    qc = qc.compose(pool_circ_1(thetas, 1, 0))
    qc = qc.compose(pool_circ_1(thetas, 3, 2))
    qc = qc.compose(pool_circ_1(thetas, 5, 4))
    qc = qc.compose(pool_circ_1(thetas, 7, 6))
    return qc

def pooling_layer_2(qc, thetas):
    qc = qc.compose(pool_circ_1(thetas, 0, 2))
    qc = qc.compose(pool_circ_1(thetas, 4, 6))
    return qc

def pooling_layer_3(qc, thetas):
    qc = qc.compose(pool_circ_1(thetas, 0, 4))
    return qc

debug = False
# Variational circuit used in below model which has parameters optimized during training
def qcnn_circ(num_qubits, layer_size = 18):
    qc = QuantumCircuit(num_qubits,1)  # just to measure the 5 th qubit (4 indexed)\
        
    thetas = ParameterVector("t", length=60)
    
    if debug == True:
        print(thetas)
    theta1 = thetas[0:layer_size]
    theta2 = thetas[layer_size: 2 * layer_size]
    theta3 = thetas[2 * layer_size: 3 * layer_size]
    theta4 = thetas[3 * layer_size: 3 * layer_size + 2]
    theta5 = thetas[3 * layer_size + 2: 3 * layer_size + 4]
    theta6 = thetas[3 * layer_size + 4: 3 * layer_size + 6]
    theta7 = thetas

    # Pooling Ansatz1 is used by default
    qc = conv_layer_1(qc, theta1)
    qc = pooling_layer_1(qc, theta4)
    qc = conv_layer_2(qc, theta2)
    qc = pooling_layer_2(qc, theta5)
    qc = conv_layer_3(qc, theta3)
    qc = pooling_layer_3(qc, theta6)

    return qc

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if p < 0.5:
            p = 0
        else:
            p = 1
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

square_loss_check = False 

# model to be used for training which has input data encoded and variational circuit is appended to it
def qcnn_model(x, num_qubits):
    
    qc = QuantumCircuit(num_qubits)
          
    # Encode the pixel data into the quantum circuit here  x is the input data which is list of 14 values   
    # feature mapping 
    for j in range(num_qubits):
        qc.rx(x[j], j )
    # Append the variational circuit ( Ansatz ) to the quantum circuit
    # thetas = theta.tolist()
    qcnn_circ_temp = qcnn_circ(num_qubits)
    qc.compose(qcnn_circ_temp, inplace=True)
    return qc


# Define the quantum instance to be used for training
backend = Aer.get_backend('qasm_simulator')

# threshold to be used for prediction of label from the circuit output
global threshold 

threshold = 0.5
    

# for training loss plot 

train_loss_history = []
train_accuracy_history = []
global num_iter_plot 


# function to calculate the loss function
def loss_function(theta, x_batch, y_batch, is_draw_circ=False, is_print=False):
    # predicted_label = []
    total_loss = 0
    epsilon = 1e-15  # small value to avoid log(0) errors
    prediction_label = []
    i_draw = 0
    for data_point, label in zip(x_batch, y_batch):
        # Create the quantum circuit for the data point
        qc = qcnn_model(data_point, num_qubits)
        qc_upd = qc.assign_parameters(theta)
        
        if i_draw==0 and is_draw_circ:
            print(qc_upd)
            i_draw += 1
        # Simulate the quantum circuit and get the result
        if expectation_calc_method == True:
            # val = expectation_calc_qcnn.calculate_expectation(qc,shots=num_shots,num_qubits=num_qubits) 
            val = exp_cal(qc_upd) 
            val=(val+1)*0.5
        prediction_label.append(float(val[0]))
        # prediction_label.append(val)
    
    # print(prediction_label)
    # loss = log_loss(y_train, prediction_label)
    if square_loss_check == True:
        loss = square_loss(y_batch, prediction_label)
    else:
        loss = log_loss(y_batch, prediction_label)

    if is_print:
        predictions = []
        for i in range(len(y_batch)):
            if prediction_label[i] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        # print(predictions)
        accuracy = accuracy_score(y_batch, predictions)
        print("Accuracy:", accuracy, "loss:", loss)
        train_loss_history.append(loss)
        train_accuracy_history.append(accuracy)
    
    return loss
# print(len(train_loss_history))

def callback(theta):
    pass
        # loss_function(theta, is_draw_circ=False, is_print=True)

# Initialize  epochs
num_epochs = 1
reps = 3

# Number of shots to run the program (experiment)
num_shots = 1000

global weights
# weights = np.random.rand(num_qubits * reps *2 )
weights = np.random.rand(60)
#weights = np.zeros(num_qubits * reps)
print(len(weights))

print("weights", weights)

# Mini Batch taining batch size 
batch_size = 35

# training samples data size 
data_size  = len(x_final_train)

# Number of batches 
batch_count =  (data_size + batch_size - 1) // batch_size

for epoch in range(num_epochs):
    
    #  Looping batch count
    for batch in range(batch_count):
      
      # start and end indeces for current_batch( Here batch is index of above batch)
        start_index = batch * batch_size   # batch will be 0 for first index
        end_index   = min((batch + 1 ) * batch_size, data_size)
      
      # batch extraction 
        x_batch = x_final_train[start_index:end_index]
        y_batch = y_train[start_index:end_index]
      
      
        print(f"Current batch is {batch}")
    # Minimize the loss using SPSA optimizer
        is_draw,is_print = False,True
        theta = minimize(loss_function, x0 = weights, args=(x_batch,y_batch,is_draw,is_print), method="COBYLA", tol=0.001, 
                                callback=callback, options={'maxiter': 100, 'disp': False} )
    #theta=SPSA(maxiter=100).minimize(loss_function, x0=weights)
        weights = theta.x
    loss = theta.fun
    print(f"Epoch {epoch+1}/{num_epochs}, loss = {loss:.4f}")


print("theta function", theta.x)
    # print(type(theta.x))

# Below code is to test the model on test data after training and getting theta values
# Perform inference with the trained QNN
predictions = []

# Loop over the test data and use theta obtained from training to get the predicted label
#print("x_scaled_test",x_scaled_test)
# print(theta.x)
# print(num_shots)
test_accuracy_history = []
for data_point in x_final_test:
    qc = qcnn_model(data_point, num_qubits)
    
    qc_upd = qc.assign_parameters(theta.x)
    # Simulate the quantum circuit and get the result
    
    # qc = qc.assign_parameters(theta.x)
    if expectation_calc_method == True:
        # val = expectation_calc_qcnn.calculate_expectation(qc,shots=num_shots,num_qubits=num_qubits)   
        val = exp_cal(qc_upd)   
        val=(val+1)*0.5
        if val > 0.5:
            predicted_label = 1
        else:
            predicted_label = 0
    print("predicted_label",predicted_label)
    predictions.append(predicted_label)
    

    # Calculate the test accuracy on the test set after each data point
    test_accuracy = accuracy_score(y_test[:len(predictions)], predictions)

    # Store the test accuracy in the history list after each data point
    test_accuracy_history.append(test_accuracy)


# Evaluate the QCNN accuracy on the test set once the model is trained and tested
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print(f"This is for {num_qubits} qubits")
print("predictions",predictions)


# training loss after each iteration
# plt.plot(range(1, num_epochs * 100 + 1), train_loss_history)
plt.plot(range(1, len(train_loss_history)+1), train_loss_history)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title(f'Training Loss History for {num_qubits} qubits')
plt.show()

# training accuracy after each iteration
plt.plot(range(1, len(train_accuracy_history)+1), train_accuracy_history)
plt.xlabel('Iteration')
plt.ylabel('Training Accuracy')
plt.title(f'Training Accuracy History for {num_qubits} qubits')
plt.show()


# testing accuracy after each data point
plt.plot(range(1, len(x_final_test) + 1), test_accuracy_history)
plt.xlabel('Data Point')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy after Each Data Point')
plt.show()