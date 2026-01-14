import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
    epsilon = 1e-20
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
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
    
    def train(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)

        # Forward pass
        outputs = self.forward(inputs)
        
        # Calcolo dell'errore
        output_errors = binary_cross_entropy_derivative(targets, outputs)
        output_deltas = output_errors * sigmoid_derivative(outputs)
        
        # Backpropagation
        hidden_errors = np.dot(output_deltas, self.weights_hidden_output.T)
        hidden_deltas = hidden_errors * sigmoid_derivative(self.hidden_layer)
        
        # Aggiornamento pesi 
        self.weights_hidden_output -= self.learning_rate * np.outer(self.hidden_layer, output_deltas)
        self.weights_input_hidden -= self.learning_rate * np.outer(inputs, hidden_deltas)
    
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

# Divisione in training e validation set (80% training, 20% validation)
train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Creare dataset per il training e validation
train_data = [(train_inputs[i], train_targets[i]) for i in range(len(train_inputs))]
val_data = [(val_inputs[i], val_targets[i]) for i in range(len(val_inputs))]

# Iperparametri
epochs = 800
hidden_size = 3
learning_rate = 0.5

# Creazione della rete neurale
nn = NeuralNetwork(input_size=17, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)

# Addestramento con SGD (online learning)
errors = []
accs = []
for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    np.random.shuffle(train_data)  # Mescoliamo i dati ad ogni epoca
    for data in train_data:
        nn.train(data[0], data[1])  # Aggiorna i pesi dopo ogni esempio

    # Calcolo dell'errore medio sul training set
    total_error = 0
    for inputs, target in train_data:
        output = nn.forward(inputs)
        total_error += np.mean(binary_cross_entropy(target, output))

    errors.append(total_error / len(train_data))

    # Validazione
    val_accuracy = nn.accuracy(val_data)
    accs.append(val_accuracy)

    if epoch % 100 == 0:  # Ogni 100 epoche, stampa la validazione
        print(f"Epoch {epoch}, Validation Accuracy: {val_accuracy:.2f}%")



# Calcolare l'accuratezza finale sul training e validation
train_accuracy = nn.accuracy(train_data)
val_accuracy = nn.accuracy(val_data)

print(f"Accuratezza sul Training Set: {train_accuracy:.2f}%")
print(f"Accuratezza sulla Validation Set: {val_accuracy:.2f}%")

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

# Visualizzazione dell'errore e della val_accuracy durante l'allenamento
plt.plot(errors)
plt.title('Errore Medio durante l\'Allenamento')
plt.xlabel('Epoca')
plt.ylabel('Errore Medio')
plt.show()
plt.plot(accs)
plt.title('Validation accuracy durante l\'allenamento')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.show()
