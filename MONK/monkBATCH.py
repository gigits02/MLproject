"""
    Full pipeline of a Neural Network for MONK dataset classification.
    
    Implemented features:
    - Three-layer neural network (input, hidden, output)
    - Sigmoid activation function
    - Binary Cross-Entropy loss
    - Batch gradient descent with backpropagation
    - Manual train/validation/test split
    - Metrics export to CSV for analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



def sigmoid(x):
    """
        Defines the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
        Defines the derivative of the sigmoid activation function.
    """
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    """
        Defines the BCE as loss function for the classification problem.
        The epsilon parameter is added to avoid divergence.
    
        :param y_true: Target value
        :param y_pred: Prediction value
    """
    y_true = np.array(y_true)
    epsilon = 1e-20
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



class NeuralNetwork:
    """
        Defines the Neural Network with:
        - 1 input layer
        - 1 hidden layer
        - 1 output layer
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Random weight initialization
        self.weights_input_hidden = np.random.uniform(-0.1, 0.1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.1, 0.1, (self.hidden_size, self.output_size))
    
    def forward(self, inputs):
        """
            Forward pass through the network.

            :param inputs: Input features
            :return: Network output
        """
        self.hidden_layer = sigmoid(np.dot(inputs, self.weights_input_hidden))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.output_layer
    
    def train(self, batch_inputs, batch_targets):
        """
            Train the network on a batch using backpropagation.
        """
        batch_inputs = np.array(batch_inputs)
        batch_targets = np.array(batch_targets).reshape(-1, 1)

        # Forward pass
        outputs = self.forward(batch_inputs)
        
        # Backpropagation - Output layer
        output_deltas = outputs - batch_targets
        
        # Backpropagation - Hidden layer
        hidden_errors = np.dot(output_deltas, self.weights_hidden_output.T)
        hidden_deltas = hidden_errors * sigmoid_derivative(self.hidden_layer)

        # Weight updates with averaged gradients
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_layer.T, output_deltas) / len(batch_inputs)
        self.weights_input_hidden -= self.learning_rate * np.dot(batch_inputs.T, hidden_deltas) / len(batch_inputs)

    def accuracy(self, dataset):
        """
            Calculate accuracy on a dataset.

            :param dataset: List of (input, target) tuples
            :return: Accuracy percentage
        """
        correct = 0
        for inputs, target in dataset:
            output = self.forward(inputs)
            predicted = np.round(output).astype(int)
            if predicted == target:
                correct += 1
        return correct / len(dataset) * 100
    
    def get_predictions(self, dataset):
        """
            Get predictions and true values for a dataset.

            :param dataset: List of (input, target) tuples
            :return: Arrays of predictions and true values
        """
        predictions = []
        true_values = []
        
        for inputs, target in dataset:
            output = self.forward(inputs.reshape(1, -1))
            predictions.append(output[0, 0])
            true_values.append(target)
        
        return np.array(predictions), np.array(true_values)


def manual_train_val_split(inputs, targets, val_size=0.2, random_seed=42):
    """
        Manual split of dataset into training and validation sets.

        :param inputs: numpy array of input features
        :param targets: numpy array of target labels
        :param val_size: proportion of data used for validation
    """
    np.random.seed(random_seed)

    n_samples = len(inputs)
    indices = np.arange(n_samples)

    # Shuffle indices
    np.random.shuffle(indices)

    # Split point
    split_idx = int(n_samples * (1 - val_size))

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_inputs = inputs[train_idx]
    val_inputs = inputs[val_idx]
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]

    return train_inputs, val_inputs, train_targets, val_targets



if __name__ == "__main__":
    # Load training data
    inputfile = "encoded_MonkFiles/m3training.csv"
    df = pd.read_csv(inputfile, header=None)

    # Separate features and targets
    inputs = df.iloc[:, 1:].values
    targets = df.iloc[:, 0].values

    # Split into training and validation sets (80% training, 20% validation)
    train_inputs, val_inputs, train_targets, val_targets = manual_train_val_split(
        inputs, targets, val_size=0.2, random_seed=42
    )

    # Create datasets
    train_data = [(train_inputs[i], train_targets[i]) for i in range(len(train_inputs))]
    val_data = [(val_inputs[i], val_targets[i]) for i in range(len(val_inputs))]

    # Hyperparameters
    epochs = 300
    batch_size = 16
    hidden_size = 4
    learning_rate = 0.5

    # Early stopping parameters
    patience = 60
    min_delta = 0

    best_val_loss = np.inf
    epochs_without_improvement = 0
    best_weights_input_hidden = None
    best_weights_hidden_output = None

    # Creating neural network
    nn = NeuralNetwork(input_size=17, hidden_size=hidden_size, output_size=1,
                        learning_rate=learning_rate)

    # Training metrics storage
    training_metrics = []

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):

        # Batch training
        for batch_start in range(0, len(train_data), batch_size):
            batch_end = min(batch_start + batch_size, len(train_data))
            batch_inputs = [train_data[i][0] for i in range(batch_start, batch_end)]
            batch_targets = [train_data[i][1] for i in range(batch_start, batch_end)]

            nn.train(batch_inputs, batch_targets)

        # Training loss
        train_preds, train_true = nn.get_predictions(train_data)
        train_loss = np.mean(binary_cross_entropy(train_true, train_preds))
        
        # Validation loss
        val_preds, val_true = nn.get_predictions(val_data)
        val_loss = np.mean(binary_cross_entropy(val_true, val_preds))
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best weights
            best_weights_input_hidden = nn.weights_input_hidden.copy()
            best_weights_hidden_output = nn.weights_hidden_output.copy()
        else:
            epochs_without_improvement += 1


        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch: {epoch}")
            break

        # Accuracies
        train_acc = nn.accuracy(train_data)
        val_acc = nn.accuracy(val_data)
        
        # Store metrics
        training_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"    Training accuracy: {train_acc:.2f}%")
            print(f"    Validation accuracy: {val_acc:.2f}%")

    final_epoch = epoch
    print(f"Training stopped at epoch: {final_epoch}")

    # Final training metrics
    train_accuracy = nn.accuracy(train_data)
    val_accuracy = nn.accuracy(val_data)

    print(f"Final Training accuracy: {train_accuracy:.2f}%")
    print(f"Final Validation accuracy: {val_accuracy:.2f}%")

    # Save training metrics to CSV
    metrics_df = pd.DataFrame(training_metrics)
    metrics_df.to_csv("monk_results/training_metrics3.csv", index=False)

    # TEST PHASE

    # Restoring best weights
    nn.weights_input_hidden = best_weights_input_hidden
    nn.weights_hidden_output = best_weights_hidden_output

    # Load and evaluate on test set
    inputfile2 = "encoded_MonkFiles/m3test.csv"
    df2 = pd.read_csv(inputfile2, header=None)

    test_inputs = df2.iloc[:, 1:].values
    test_targets = df2.iloc[:, 0].values

    test_data = [(test_inputs[i], test_targets[i]) for i in range(len(test_inputs))]

    test_preds, test_true = nn.get_predictions(test_data)
    test_loss = np.mean(binary_cross_entropy(test_true, test_preds))
    test_accuracy = nn.accuracy(test_data)

    print(f"Test accuracy: {test_accuracy:.2f}%")

    run_summary = {
        "dataset": "MONK-3",
        "epochs_planned": epochs,
        "epochs_run": final_epoch + 1,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss
    }

    summary_df = pd.DataFrame([run_summary])

    summary_file = "monk_results/Monk3_summary.csv"

    # Append if file exists
    try:
        existing_df = pd.read_csv(summary_file)
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    except FileNotFoundError:
        pass

    summary_df.to_csv(summary_file, index=False)



    # Plotting training and validation errors
    plt.figure(figsize=(12, 4))

    plt.plot(
        metrics_df["epoch"].to_numpy(),
        metrics_df["train_loss"].to_numpy(),
        label="Train Loss"
    )
    plt.plot(
        metrics_df["epoch"].to_numpy(),
        metrics_df["val_loss"].to_numpy(),
        label="Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (MONK-3)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
