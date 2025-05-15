import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Funzione di attivazione (Sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# Binary Cross Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    # Aggiunta epsilon per evitare log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Derivata della BCE rispetto a y_pred (output della rete)
def binary_cross_entropy_derivative(y_true, y_pred):
    y_true = np.array(y_true)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        
        self.learning_rate = 0.1
    
    def forward(self, inputs):
        self.hidden_layer = sigmoid(np.dot(inputs, self.weights_input_hidden))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.output_layer
    
    def train(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Calcolo dell'errore con BCE
        output_errors = binary_cross_entropy_derivative(targets, outputs)
        output_deltas = output_errors * sigmoid_derivative(outputs)
        
        # Backpropagation
        hidden_errors = np.dot(output_deltas, self.weights_hidden_output.T)
        hidden_deltas = hidden_errors * sigmoid_derivative(self.hidden_layer)
        
        # Aggiornamento pesi
        self.weights_hidden_output -= self.learning_rate * np.outer(self.hidden_layer, output_deltas)
        self.weights_input_hidden -= self.learning_rate * np.outer(inputs, hidden_deltas)


# Esempio di utilizzo (XOR)
nn = NeuralNetwork(2, 4, 1)
training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

# Per tracciare l'errore medio e l'accuratezza
errors = []
accs = []
epochs = 50000

for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    data = training_data[np.random.randint(len(training_data))]
    nn.train(data[0], data[1])
    
    # Calcolo dell'errore medio e accuratezza
    total_error = 0
    correct = 0
    for inputs, target in training_data:
        output = nn.forward(inputs)
        total_error += np.mean(binary_cross_entropy(target, output))
        predicted = np.round(output).astype(int)
        if np.array_equal(predicted, target):
            correct += 1
    
    errors.append(total_error / len(training_data))
    accs.append(correct / len(training_data) * 100)

# Test finale
for data in training_data:
    print(f"Input: {data[0]}, Output: {nn.forward(data[0])}")

print(f"Accuratezza finale: {accs[-1]:.2f}%")
print(f"Loss finale: {errors[-1]:.4f}")

# Grafici
plt.plot(errors)
plt.title("Errore BCE durante l'allenamento")
plt.xlabel("Epoca")
plt.ylabel("Errore Medio")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.plot(accs)
plt.title("Accuracy durante l'allenamento")
plt.xlabel("Epoca")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
