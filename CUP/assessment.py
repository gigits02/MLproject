import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time

# Genera un seed casuale basato sul tempo corrente
seed = int(time.time())  # Usa il tempo corrente per creare un seed casuale
random.seed(seed)
# Stampa il seed
print('RUNNING SEED: ', seed)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def normalize_data(X):
    X = np.array(X)
    min_values = X.min(axis=0)
    max_values = X.max(axis=0)
    return (X - min_values) / (max_values - min_values + 10**(-8))  # Evitare divisioni per zero

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    inputs = data.iloc[:, 1:13].values  # 12 colonne centrali come input
    targets = data.iloc[:, 13:].values  # Ultime 3 colonne come target
    return inputs, targets

# Divisione in training, validation e test set
def split_data(inputs, targets, train_ratio=0.6, val_ratio=0.2):

    num_samples = inputs.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Calcola le dimensioni per i vari set
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    # Indici per i vari set
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return inputs[train_idx], targets[train_idx], inputs[val_idx], targets[val_idx], inputs[test_idx], targets[test_idx]

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, eta0, l2_lambda, alpha):
        
        self.initial_learning_rate = eta0
        self.learning_rate = eta0
        self.l2_lambda = l2_lambda
        self.alpha = alpha

        self.network = {
            'W1': np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(1 / input_size),
            'W2': np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(1 / hidden_sizes[0]),
            'W3': np.random.randn(hidden_sizes[1], output_size) * np.sqrt(1 / hidden_sizes[1]),
        }
        
        # Velocità per il momentum di Nesterov
        self.velocities = {
            'vW1': np.zeros_like(self.network['W1']),
            'vW2': np.zeros_like(self.network['W2']),
            'vW3': np.zeros_like(self.network['W3']),
        }

    def compute_loss(self, predictions, targets):

        mse_loss = np.mean((predictions - targets) ** 2)
        l2_loss = (self.l2_lambda / 2) * (np.sum(self.network['W1']**2) +
                                          np.sum(self.network['W2']**2) +
                                          np.sum(self.network['W3']**2))
        
        return mse_loss + l2_loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        train_losses = []
        val_losses = []
        num_samples = X_train.shape[0]
        
        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                # Anticipazione del momentum (Nesterov)
                for key in self.network:
                    self.network[key] += self.alpha * self.velocities['v' + key]
                
                forward_results = self.forward_propagation(X_batch)
                
                # Gradiente MSE per l'output
                dZ3 = 2 * (forward_results['Z3'] - y_batch) / y_batch.shape[0]
                # Backpropagation con regolarizzazione L2
                dW3 = forward_results['A2'].T @ dZ3 + self.l2_lambda * self.network['W3']
                dA2 = dZ3 @ self.network['W3'].T
                dZ2 = dA2 * relu_derivative(forward_results['Z2'])
                dW2 = forward_results['A1'].T @ dZ2 + self.l2_lambda * self.network['W2']
                dA1 = dZ2 @ self.network['W2'].T
                dZ1 = dA1 * relu_derivative(forward_results['Z1'])
                dW1 = X_batch.T @ dZ1 + self.l2_lambda * self.network['W1']
                
                # Aggiornamento velocità (momentum di Nesterov)
                self.velocities['vW3'] = self.alpha * self.velocities['vW3'] - self.learning_rate * dW3
                self.velocities['vW2'] = self.alpha * self.velocities['vW2'] - self.learning_rate * dW2
                self.velocities['vW1'] = self.alpha * self.velocities['vW1'] - self.learning_rate * dW1
                
                # Aggiornamento pesi
                self.network['W3'] += self.velocities['vW3']
                self.network['W2'] += self.velocities['vW2']
                self.network['W1'] += self.velocities['vW1']

                # DA METTERE FUORI DAL CICLO SUI BATCH O CONSIDERARE I BATCH NELLA FORMULA DEL DECAY?
                # Learning rate decay
                #self.learning_rate = self.initial_learning_rate * (1 - epoch / epochs)
                #self.learning_rate = max(self.learning_rate, 1e-6)  # Per evitare che diventi troppo piccolo
            

            # Calcolo delle loss
            train_loss = self.compute_loss(self.predict(X_train), y_train)
            train_losses.append(train_loss)
            
            val_predictions = self.predict(X_val)
            val_loss = self.compute_loss(val_predictions, y_val)
            val_losses.append(val_loss)

        print(f"\nFinal Train Loss: {train_losses[-1]:.6f}")
        print(f"Final Validation Loss: {val_losses[-1]:.6f}")        

        self.plot_losses(train_losses, val_losses)
        return train_losses, val_losses

    def forward_propagation(self, X):
        Z1 = X @ self.network['W1']
        A1 = relu(Z1)

        Z2 = A1 @ self.network['W2']
        A2 = relu(Z2)
        
        Z3 = A2 @ self.network['W3'] # Nessuna attivazione per l'output

        return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3}
    
    def predict(self, X):
        return self.forward_propagation(X)['Z3']

    def plot_losses(self, train_losses, val_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.show()
#----------------------------------------------------------------------------------------------

# Caricamento e preparazione dati
inputs, targets = load_data('cupTrain.csv')

# Divisione in training, validation e test set
X_train, y_train, X_val, y_val, X_test, y_test = split_data(inputs, targets)

#X_train = normalize_data(X_train)
#X_val = normalize_data(X_val)
#X_test = normalize_data(X_test)

# Parametri di allenamento
epochs = 500
batch_size = 30

# Parametri della rete
input_size = X_train.shape[1]
hidden_sizes = [60, 60] 
output_size = y_train.shape[1]
eta0 = 0.001 / batch_size
l2_lambda = 0.00005 
alpha = 0.9

#Inizializzazione
nn = NeuralNetwork(input_size, hidden_sizes, output_size, eta0, l2_lambda, alpha)
#Allenamento
train_losses = []
val_losses = []
train_losses, val_losses = nn.train(X_train, y_train, X_val, y_val, epochs, batch_size)

#---------------------------------------------------------------------------------------------------

# Plotta i risultati (training)
y_pred_tr = nn.predict(X_train)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_train[:, 0], y_train[:, 1], y_train[:, 2], c='blue', label='Real Targets', alpha=0.6)
ax.scatter(y_pred_tr[:, 0], y_pred_tr[:, 1], y_pred_tr[:, 2], c='red', label='Predicted Points', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('TRAINING')
ax.legend()
plt.show()
# Plotta i risultati (validation)
y_pred_val = nn.predict(X_val)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_val[:, 0], y_val[:, 1], y_val[:, 2], c='blue', label='Real Targets', alpha=0.6)
ax.scatter(y_pred_val[:, 0], y_pred_val[:, 1], y_pred_val[:, 2], c='red', label='Predicted Points', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('VALIDATION')
ax.legend()
plt.show()
# Plotta i risultati predetti con tr e val data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_pred_tr[:, 0], y_pred_tr[:, 1], y_pred_tr[:, 2], c='blue', label='Predicted TR', alpha=0.6)
ax.scatter(y_pred_val[:, 0], y_pred_val[:, 1], y_pred_val[:, 2], c='red', label='Predicted VL', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('COMPARING PREDICTIONS')
ax.legend()
plt.show()
# TEST DELLA RETE
y_pred_ts = nn.predict(X_test)
test_loss = nn.compute_loss(y_pred_ts, y_test)
print(f"Test Loss: {test_loss:.6f}")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_test[:, 0], y_test[:, 1], y_test[:, 2], c='blue', label='TS Targets', alpha=0.6)
ax.scatter(y_pred_ts[:, 0], y_pred_ts[:, 1], y_pred_ts[:, 2], c='red', label='Predicted Points', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('TEST')
ax.legend()
plt.show()
#BLIND TEST
tsData = pd.read_csv('cupTest.csv', header=None)
X_test_blind = tsData.iloc[:, 1:].values
y_pred_blind = nn.predict(X_test_blind)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_pred_blind[:, 0], y_pred_blind[:, 1], y_pred_blind[:, 2], c='red', label='Predicted Points', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('BLIND TEST')
ax.legend()
plt.show()

