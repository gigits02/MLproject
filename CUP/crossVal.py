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

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    inputs = data.iloc[:, 1:13].values  # 12 colonne centrali come input
    targets = data.iloc[:, 13:].values  # Ultime 4 colonne come target
    return inputs, targets

# Divisione in training+CV e test set
def train_test_split(X, y, test_ratio=0.2):
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def plot_component_comparison(y_true, y_pred, title_prefix=""):
    num_outputs = y_true.shape[1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(num_outputs):
        axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        
        # retta y = x (predizione perfetta)
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axes[i].set_xlabel(f'Target {i+1}')
        axes[i].set_ylabel(f'Prediction {i+1}')
        axes[i].set_title(f'{title_prefix} Output {i+1}')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_error_per_component(y_true, y_pred, title="Error per output"):
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 5))
    plt.boxplot(errors, tick_labels=[f'Output {i+1}' for i in range(errors.shape[1])])
    plt.ylabel('Prediction Error')
    plt.title(title)
    plt.grid(True)
    plt.show()


def k_fold_split(X, y, k):
    indices = np.random.permutation(len(X))
    folds = np.array_split(indices, k)

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, eta0, l2_lambda, alpha):
        
        self.initial_learning_rate = eta0
        self.learning_rate = eta0
        self.l2_lambda = l2_lambda
        self.alpha = alpha

        # Inizializzazione Xavier/Glorot uniforme
        fan_in_1, fan_out_1 = input_size, hidden_sizes[0]
        fan_in_2, fan_out_2 = hidden_sizes[0], hidden_sizes[1]
        fan_in_3, fan_out_3 = hidden_sizes[1], output_size

        limit1 = np.sqrt(6 / (fan_in_1 + fan_out_1))
        limit2 = np.sqrt(6 / (fan_in_2 + fan_out_2))
        limit3 = np.sqrt(6 / (fan_in_3 + fan_out_3))

        self.network = {
            'W1': np.random.uniform(-limit1, limit1, (fan_in_1, fan_out_1)),
            'b1': np.zeros(hidden_sizes[0]),
            'W2': np.random.uniform(-limit2, limit2, (fan_in_2, fan_out_2)),
            'b2': np.zeros(hidden_sizes[1]),
            'W3': np.random.uniform(-limit3, limit3, (fan_in_3, fan_out_3)),
            'b3': np.zeros(output_size)
        }
        
        # Velocità per il momentum di Nesterov
        self.velocities = {
            'vW1': np.zeros_like(self.network['W1']),
            'vW2': np.zeros_like(self.network['W2']),
            'vW3': np.zeros_like(self.network['W3']),
            'vb1': np.zeros_like(self.network['b1']),
            'vb2': np.zeros_like(self.network['b2']),
            'vb3': np.zeros_like(self.network['b3']),
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
                dZ2 = dA2 * tanh_derivative(forward_results['Z2'])
                dW2 = forward_results['A1'].T @ dZ2 + self.l2_lambda * self.network['W2']
                dA1 = dZ2 @ self.network['W2'].T
                dZ1 = dA1 * tanh_derivative(forward_results['Z1'])
                dW1 = X_batch.T @ dZ1 + self.l2_lambda * self.network['W1']
                
                # Gradiente dei bias
                db3 = np.sum(dZ3, axis=0)
                db2 = np.sum(dZ2, axis=0)
                db1 = np.sum(dZ1, axis=0)

                # Aggiornamento velocità (momentum di Nesterov)
                self.velocities['vW3'] = self.alpha * self.velocities['vW3'] - self.learning_rate * dW3
                self.velocities['vW2'] = self.alpha * self.velocities['vW2'] - self.learning_rate * dW2
                self.velocities['vW1'] = self.alpha * self.velocities['vW1'] - self.learning_rate * dW1
                
                # Aggiornamento pesi
                self.network['W3'] += self.velocities['vW3']
                self.network['W2'] += self.velocities['vW2']
                self.network['W1'] += self.velocities['vW1']


                # Aggiornamento velocità bias
                self.velocities['vb3'] = self.alpha * self.velocities['vb3'] - self.learning_rate * db3
                self.velocities['vb2'] = self.alpha * self.velocities['vb2'] - self.learning_rate * db2
                self.velocities['vb1'] = self.alpha * self.velocities['vb1'] - self.learning_rate * db1

                # Aggiornamento bias
                self.network['b3'] += self.velocities['vb3']
                self.network['b2'] += self.velocities['vb2']
                self.network['b1'] += self.velocities['vb1']

                # Learning rate linear decay
                #self.learning_rate = self.initial_learning_rate * (1 - epoch / epochs)
                #self.learning_rate = max(self.learning_rate, 1e-5)  # Per evitare che diventi troppo piccolo
                # Learning rate exp decay 
                # self.learning_rate = self.initial_learning_rate * (0.99 ** epoch)

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
        Z1 = X @ self.network['W1'] + self.network['b1']
        A1 = tanh(Z1)

        Z2 = A1 @ self.network['W2'] + self.network['b2']
        A2 = tanh(Z2)
        
        Z3 = A2 @ self.network['W3'] + self.network['b3'] # Nessuna attivazione per l'output

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
inputs, targets = load_data('./CUP_datasets/ML-CUP25-TR.csv')

# Divisione in training e test set
X_train_full, y_train_full, X_test, y_test = train_test_split(inputs, targets, test_ratio=0.2)

# Parametri di allenamento
epochs = 1000
batch_size = 30
k = 5

# Parametri della rete
hidden_sizes = [20, 20] 
eta0 = 0.0005
l2_lambda = 0.0001
alpha = 0.9

train_losses = []
val_losses = []

for fold, (X_tr, y_tr, X_val, y_val) in enumerate(k_fold_split(X_train_full, y_train_full, k=k)):
    print(f"\nFold {fold+1}/{k}")

    # Standardizzazione sul training fold (input e target)
    X_mean, X_std = X_tr.mean(axis=0), X_tr.std(axis=0)
    y_mean, y_std = y_tr.mean(axis=0), y_tr.std(axis=0)
    X_tr = (X_tr - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    y_tr = (y_tr - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    nn = NeuralNetwork(
        input_size=X_tr.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=y_tr.shape[1],
        eta0=eta0,
        l2_lambda=l2_lambda,
        alpha=alpha
    )

    train_loss, val_loss = nn.train(X_tr, y_tr, X_val, y_val, epochs, batch_size)
    train_losses.append(train_loss[-1])
    val_losses.append(val_loss[-1])

print(f"\nCV Train Loss: {np.mean(train_losses):.6f} ± {np.std(train_losses):.6f}")
print(f"\nCV Validation Loss: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")

# ----------------------------
# Predizioni e destandardizzazione
# ----------------------------

# Standardizzazione su tutto il training
X_mean, X_std = X_train_full.mean(axis=0), X_train_full.std(axis=0)
y_mean, y_std = y_train_full.mean(axis=0), y_train_full.std(axis=0)
X_train_std = (X_train_full - X_mean) / X_std
X_test_std = (X_test - X_mean) / X_std
y_train_std = (y_train_full - y_mean) / y_std
y_test_std = (y_test - y_mean) / y_std

# TRAIN
y_pred_tr = nn.predict(X_train_std)
y_pred_tr_dest = y_pred_tr * y_std + y_mean
real_tr_loss = nn.compute_loss(y_pred_tr_dest, y_train_full)
print("Original scale Loss (entire tr-set):", real_tr_loss)
plot_component_comparison(y_train_full, y_pred_tr_dest, title_prefix="TRAIN")
plot_error_per_component(y_train_full, y_pred_tr_dest, title="TRAIN Error Distribution")

# TEST
y_pred_test = nn.predict(X_test_std) * y_std + y_mean