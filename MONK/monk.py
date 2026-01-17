import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time

seed = 1768656537
#seed = int(time.time()) 
random.seed(seed)
np.random.seed(seed)
# Stampa il seed
print('RUNNING SEED: ', seed)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-20
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# -------------------------------
# Utility
# -------------------------------

def train_val_split(X, y, ratio):
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def compute_loss(model, X, y):
    y = y.reshape(-1, 1)
    outputs = model.forward(X)
    bce = np.mean(binary_cross_entropy(y, outputs))

    l2_term = 0.5 * model.lambda_l2 * (
        np.sum(model.weights_input_hidden**2) +
        np.sum(model.weights_hidden_output**2)
    )

    return bce + l2_term

# -------------------------------
# Early stopping
# -------------------------------

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights_input_hidden = None
        self.best_weights_hidden_output = None
        self.should_stop = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights_input_hidden = model.weights_input_hidden.copy()
            self.best_weights_hidden_output = model.weights_hidden_output.copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

    def restore_best_weights(self, model):
        model.weights_input_hidden = self.best_weights_input_hidden
        model.weights_hidden_output = self.best_weights_hidden_output

# -------------------------------
# Neural Network
# -------------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, lambda_l2):
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2

        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

    def forward(self, inputs):
        self.hidden_layer = sigmoid(np.dot(inputs, self.weights_input_hidden))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.output_layer

    def train(self, batch_inputs, batch_targets):
        batch_inputs = np.array(batch_inputs)
        batch_targets = np.array(batch_targets).reshape(-1, 1)

        outputs = self.forward(batch_inputs)

        output_deltas = outputs - batch_targets
        hidden_errors = np.dot(output_deltas, self.weights_hidden_output.T)
        hidden_deltas = hidden_errors * sigmoid_derivative(self.hidden_layer)

        # Gradienti standard
        grad_hidden_output = np.dot(self.hidden_layer.T, output_deltas) / len(batch_inputs)
        grad_input_hidden = np.dot(batch_inputs.T, hidden_deltas) / len(batch_inputs)

        # 🔹 L2 regularization (weight decay)
        grad_hidden_output += self.lambda_l2 * self.weights_hidden_output
        grad_input_hidden += self.lambda_l2 * self.weights_input_hidden

        # Update
        self.weights_hidden_output -= self.learning_rate * grad_hidden_output
        self.weights_input_hidden -= self.learning_rate * grad_input_hidden


    def accuracy(self, dataset):
        correct = 0
        for inputs, target in dataset:
            output = self.forward(inputs.reshape(1, -1))
            predicted = np.round(output).astype(int)
            if predicted == target:
                correct += 1
        return correct / len(dataset) * 100

# -------------------------------
# Load data
# -------------------------------

df = pd.read_csv("./encoded_MonkFiles/m3training.csv", header=None)

inputs = df.iloc[:, 1:].values
targets = df.iloc[:, 0].values

train_inputs, train_targets, val_inputs, val_targets = train_val_split(inputs, targets, ratio=0.2)

train_data = [(train_inputs[i], train_targets[i]) for i in range(len(train_inputs))]
val_data = [(val_inputs[i], val_targets[i]) for i in range(len(val_inputs))]

df_test = pd.read_csv("./encoded_MonkFiles/m3test.csv", header=None)
test_inputs = df_test.iloc[:, 1:].values
test_targets = df_test.iloc[:, 0].values
test_data = [(test_inputs[i], test_targets[i]) for i in range(len(test_inputs))]

# -------------------------------
# Hyperparameters
# -------------------------------

epochs = 1000
batch_size = 15
hidden_size = 4
learning_rate = 0.1
lambda_l2 = 0.005
nn = NeuralNetwork(input_size=17, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate, lambda_l2=lambda_l2)
early_stopping = EarlyStopping(patience=20, min_delta=0.005)

# -------------------------------
# Training
# -------------------------------

tr_loss, val_loss, ts_loss = [], [], []
tr_accs, val_accs, ts_accs = [], [], []

for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):

    for batch_start in range(0, len(train_data), batch_size):
        batch_end = min(batch_start + batch_size, len(train_data))
        batch_inputs = [train_data[i][0] for i in range(batch_start, batch_end)]
        batch_targets = [train_data[i][1] for i in range(batch_start, batch_end)]
        nn.train(batch_inputs, batch_targets)

    tr_loss.append(compute_loss(nn, train_inputs, train_targets))
    val_loss.append(compute_loss(nn, val_inputs, val_targets))
    ts_loss.append(compute_loss(nn, test_inputs, test_targets))
    tr_accs.append(nn.accuracy(train_data))
    val_accs.append(nn.accuracy(val_data))
    ts_accs.append(nn.accuracy(test_data))

    if epoch > 100:
        early_stopping.step(val_loss[-1], nn)
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch} "
                f"(best val loss = {early_stopping.best_loss:.5f})")
            break

early_stopping.restore_best_weights(nn)

# -------------------------------
# Final evaluation
# -------------------------------

print(f"Accuratezza Training: {tr_accs[-1]:.2f}%")
print(f"Accuratezza Validation: {val_accs[-1]:.2f}%")
print(f"Accuratezza Test: {ts_accs[-1]:.2f}%")


# -------------------------------
# Plot
# -------------------------------

plt.plot(tr_loss, label="Train Loss")
plt.plot(val_loss, linestyle="--", label="Val Loss")
plt.plot(ts_loss, linestyle="--", label="Ts Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy (reg.)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.title("Monk 3 (reg.) - Loss", fontweight="bold")
plt.tight_layout()
plt.show()

plt.plot(tr_accs, label="Train Accuracy")
plt.plot(val_accs, linestyle="--", label="Val Accuracy")
plt.plot(ts_accs, linestyle="--", label="Ts Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.title("Monk 3 (reg.) - Accuracy", fontweight="bold")
plt.tight_layout()
plt.show()
