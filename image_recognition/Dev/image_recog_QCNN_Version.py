import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def fetch_filter_mnist_data():
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
    return x, y

def preprocess_data(x_train, x_test,y_train, y_test):
    # Reduce the size of the dataset for faster execution
    x_train = x_train[:200]
    y_train = y_train[:200]
    x_test = x_test[:50]
    y_test = y_test[:50]

    # Number of qubits for the quantum circuit
    num_qubits = 8

    # PCA Compression
    pca = PCA(n_components=num_qubits)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Apply min-max scaling to the data to bring the values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaled_train = scaler.fit_transform(x_train_pca)
    x_scaled_test = scaler.transform(x_test_pca)

    return x_scaled_train, x_scaled_test,y_train, y_test

# ------- Below implementation is referenced from qiskit tutorial on image classification using quantum neural networks---------
    
def define_circuits(feature_map, ansatz):
      
    num_qubits = 8

    # Define the convolutional quantum circuit
    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    # Define the convolutional layer
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index: (param_index + 3)]), [q1, q2])
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index: (param_index + 3)]), [q1, q2])
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    # Define the pooling quantum circuit
    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)

        return target

    # Define the pooling layer
    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index: (param_index + 3)]), [source, sink])
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc


    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine the feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    return circuit

def callback_graph(weights, obj_func_eval):
    # clear_output(wait=True)
    # objective_func_vals.append(obj_func_eval)
    # plt.title("Objective function value against iteration")
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective function value")
    # plt.plot(range(len(objective_func_vals)), objective_func_vals)
    # plt.show()
    pass


def main():
    
    x, y = fetch_filter_mnist_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_scaled_train, x_scaled_test ,y_train,y_test = preprocess_data(x_train, x_test,y_train, y_test)
    
    # Define the feature map to encode the data
    feature_map = ZFeatureMap(8)

    # Define the ansatz to use classical optimization to find the optimal parameters
    ansatz = QuantumCircuit(8, name="Ansatz")
    
    # Define the quantum circuit 
    circuit = define_circuits(feature_map, ansatz)

    # calculate the expectation value of the observable
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # initialize the parameters
    np.random.seed(0)
    initial_point = np.random.rand(63)


    # Dclare the QNN with the circuit, observable, input parameters and weight parameters
    qcnn =  EstimatorQNN(circuit=circuit.decompose(),observables=observable,input_params=feature_map.parameters,weight_params=ansatz.parameters,)

    # Classifier with the QCNN and optimizer and initial weights
    classifier = NeuralNetworkClassifier(
        qcnn,
        optimizer=COBYLA(maxiter=50),
        callback=callback_graph,
        initial_point=initial_point,
    )

#  Testing the training data on the classifier
    x_train_final = np.asarray(x_scaled_train.tolist())
    y_train_final = np.asarray(y_train.tolist())
    
    classifier.fit(x_train_final, y_train_final)
    # Score the classifier
    accuracy = 100 * classifier.score(x_train_final, y_train_final)
    print(f"Accuracy from the train data: {np.round(accuracy, 2)}%")
    

# Testing the test data on the classifier
    x_test_final = np.asarray(x_scaled_test.tolist())
    y_test_final = np.asarray(y_test.tolist())

    # Score the classifier
    y_predict = classifier.predict(x_test_final)
    x = np.asarray(x_test_final)
    y = np.asarray(y_test_final)
    print(f"Accuracy from the test data : {np.round(100 * classifier.score(x_test_final, y_test_final), 2)}%")
    

# call the main function
if __name__ == "__main__":
    main()
