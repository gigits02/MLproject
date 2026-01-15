import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Funzione di attivazione (Sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    # Aggiunta epsilon per evitare log(0)
    epsilon = 1e-20
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    y_true = np.array(y_true)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def train_val_split(X, y, ratio):
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

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
        """
        Valuta la validation loss e decide se fermare il training.
        Salva automaticamente i pesi migliori.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            # Salvataggio pesi
            self.best_weights_input_hidden = model.weights_input_hidden.copy()
            self.best_weights_hidden_output = model.weights_hidden_output.copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

    def restore_best_weights(self, model):
        """Ripristina i pesi migliori salvati"""
        model.weights_input_hidden = self.best_weights_input_hidden
        model.weights_hidden_output = self.best_weights_hidden_output

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inizializzazione pesi
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
    
    def forward(self, inputs):
        self.hidden_layer = sigmoid(np.dot(inputs, self.weights_input_hidden))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.output_layer
    
    def train(self, batch_inputs, batch_targets):
        batch_inputs = np.array(batch_inputs)
        batch_targets = np.array(batch_targets).reshape(-1, 1)

        # Forward pass
        outputs = self.forward(batch_inputs)
        
        # Calcolo dell'errore
        output_errors = binary_cross_entropy_derivative(batch_targets, outputs)
        output_deltas = output_errors * sigmoid_derivative(outputs)
        
        # Backpropagation
        hidden_errors = np.dot(output_deltas, self.weights_hidden_output.T)
        hidden_deltas = hidden_errors * sigmoid_derivative(self.hidden_layer)

        # Aggiornamento pesi
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_layer.T, output_deltas) / len(batch_inputs)
        self.weights_input_hidden -= self.learning_rate * np.dot(batch_inputs.T, hidden_deltas) / len(batch_inputs)

    def accuracy(self, dataset):
        correct = 0
        for inputs, target in dataset:
            output = self.forward(inputs)
            predicted = np.round(output).astype(int)
            if predicted == target:
                correct += 1
        return correct / len(dataset) * 100

# Caricare i dati dal file CSV
filename = "./encoded_MonkFiles/m3training.csv"
df = pd.read_csv(filename, header=None)

# Separare input e target
inputs = df.iloc[:, 1:].values
targets = df.iloc[:, 0].values

# Divisione in training e test set (80% training, 20% validation)
train_inputs, train_targets, val_inputs, val_targets = train_val_split(inputs, targets, ratio=0.2)

# Creare dataset per il training e validation
train_data = [(train_inputs[i], train_targets[i]) for i in range(len(train_inputs))]
val_data = [(val_inputs[i], val_targets[i]) for i in range(len(val_inputs))]

# Iperparametri
epochs = 500
batch_size = 15
hidden_size = 3
learning_rate = 0.5

# Creazione della rete neurale
nn = NeuralNetwork(input_size=17, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)
early_stopping = EarlyStopping(patience=20,min_delta=0.005)


# Addestramento
tr_loss = []
val_loss = []
tr_accs = []
val_accs = []
for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    
    # Itera su tutti i batch
    for batch_start in range(0, len(train_data), batch_size):
        batch_end = min(batch_start + batch_size, len(train_data))  # Assicuro di non superare la lunghezza
        batch_inputs = [train_data[i][0] for i in range(batch_start, batch_end)]
        batch_targets = [train_data[i][1] for i in range(batch_start, batch_end)]
        
        nn.train(batch_inputs, batch_targets)

    # Calcolo dell'errore medio sul training e val set
    tr_error = 0
    for inputs, target in train_data:
        output = nn.forward(inputs)
        tr_error += np.mean(binary_cross_entropy(target, output))
    tr_loss.append(tr_error / len(train_data))
    val_error = 0
    for inputs, target in val_data:
        output = nn.forward(inputs)
        val_error += np.mean(binary_cross_entropy(target, output))
    val_loss.append(val_error / len(val_data))
    
    # Calcolo dell'accuracy media sul training e val set
    tr_accuracy = nn.accuracy(train_data)
    val_accuracy = nn.accuracy(val_data)
    tr_accs.append(tr_accuracy)
    val_accs.append(val_accuracy)
    
    if epoch % 100 == 0:  # Ogni 100 epoche, stampa la validazione
        print(f"Epoch {epoch}, Validation Accuracy: {val_accuracy:.2f}%")


    # Check early stopping
    current_val_loss = val_loss[-1]
    early_stopping.step(current_val_loss, nn)

    if early_stopping.should_stop:
        print(f"\nEarly stopping at epoch {epoch} "
        f"(best val loss = {early_stopping.best_loss:.5f})")
        break


early_stopping.restore_best_weights(nn)


# Visualizza accuratezza finale su training e validation
print(f"Accuratezza sul Training Set: {tr_accs[-1]:.2f}%")
print(f"Accuratezza sulla Validation Set: {val_accs[-1]:.2f}%")

#TEST DELLA RETE
# Caricare i dati di testing dal file CSV
filename2 = "./encoded_MonkFiles/m3test.csv"
df2 = pd.read_csv(filename2, header=None)

# Separare input e target per il test dataset
test_inputs = df2.iloc[:, 1:].values
test_targets = df2.iloc[:, 0].values

# Creare dataset per il test
test_data = [(test_inputs[i], test_targets[i]) for i in range(len(test_inputs))]

test_accuracy = nn.accuracy(test_data)

print(f"Accuratezza sul Test Set: {test_accuracy:.2f}%")

# Visualizzazione di loss e accuracy durante l'allenamento
plt.plot(tr_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy')
plt.legend()
plt.title('Losses over Epochs')
plt.show()
plt.plot(tr_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Accuracies over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
