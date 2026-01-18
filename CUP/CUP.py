import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime

class NeuralNetwork:
    """
        Neural Network implementation from scratch for multi-target regression.
        Supports multiple hidden layers with customizable activation functions.
    """
    def __init__(self, layer_sizes, learning_rate=0.01, 
                 momentum=0.0, lambda_reg=0.0,
                 activation='tanh'):
        """
            Args:
            layer_sizes: list with number of units per layer [input, hidden1, ..., output]
            learning_rate: learning rate (eta)
            momentum: momentum coefficient (alpha)
            lambda_reg: L2 regularization parameter
            activation: activation function ("sigmoid", "tanh", "relu")
        """
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.momentum = momentum
        self.lambda_reg = lambda_reg
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.velocity_w = []
        self.velocity_b = []
        
        # Xavier/He initialization
        for i in range(len(layer_sizes) - 1):
            if activation == 'relu':
                # He initialization
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier initialization
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
            self.velocity_w.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))
    
    def _activation_function(self, x):
        """
            Applies activation function
        """
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        return x
    
    def _activation_derivative(self, x):
        """
            Derivatives of activation function
        """
        if self.activation == 'sigmoid':
            s = self._activation_function(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        return np.ones_like(x)
    
    def forward(self, X):
        """
            Forward propagation
            Returns: activations and net inputs for each layer
        """
        activations = [X]
        nets = []
        
        for i in range(len(self.weights)):
            net = activations[-1] @ self.weights[i] + self.biases[i]
            nets.append(net)
            
            # Linear activation for output layer, else use specified activation
            if i == len(self.weights) - 1:
                act = net
            else:
                act = self._activation_function(net)
            
            activations.append(act)
        
        return activations, nets
    
    def backward(self, X, y, activations, nets):
        """
            Backpropagation with gradient descent
        """
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer delta (MSE loss derivative)
        deltas[-1] = (activations[-1] - y)
        
        # Hidden layers deltas
        for i in range(len(self.weights) - 2, -1, -1):
            delta_next = deltas[i + 1] @ self.weights[i + 1].T
            deltas[i] = delta_next * self._activation_derivative(nets[i])
        
        # Compute weight and bias gradients
        weight_grads = []
        bias_grads = []
        
        for i in range(len(self.weights)):
            weight_grad = (activations[i].T @ deltas[i]) / m
            bias_grad = np.sum(deltas[i], axis=0, keepdims=True) / m

            # Add L2 regularization
            if self.lambda_reg > 0:
                weight_grad += (self.lambda_reg / m) * self.weights[i]
            
            weight_grads.append(weight_grad)
            bias_grads.append(bias_grad)
        
        return weight_grads, bias_grads
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=1000, batch_size=None,
              early_stopping_patience=50, verbose=False):
        """
            Training the network with Nesterov
            Returns: Dictionary with training history
        """
        history = {'train_mse': [], 'val_mse': [], 'train_mee': [], 'val_mee': []}
        best_val_mse = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        if batch_size is None:
            batch_size = len(X_train)
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            for start_idx in range(0, len(X_train), batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Compute gradients at lookahead position
                activations, nets = self.forward(X_batch)
                weight_grads, bias_grads = self.backward(X_batch, y_batch, activations, nets)

                # Updating weights with Nesterov
                for i in range(len(self.weights)):
                    prev_vw = self.velocity_w[i]
                    prev_vb = self.velocity_b[i]

                    self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.lr * weight_grads[i]
                    self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.lr * bias_grads[i]

                    self.weights[i] += -self.momentum * prev_vw + (1 + self.momentum) * self.velocity_w[i]
                    self.biases[i]  += -self.momentum * prev_vb + (1 + self.momentum) * self.velocity_b[i]

            
            # Calculate metrics
            train_pred = self.predict(X_train)
            train_mse = np.mean((train_pred - y_train) ** 2)
            train_mee = np.mean(np.sqrt(np.sum((train_pred - y_train) ** 2, axis=1)))
            
            history['train_mse'].append(train_mse)
            history['train_mee'].append(train_mee)
            
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_mse = np.mean((val_pred - y_val) ** 2)
                val_mee = np.mean(np.sqrt(np.sum((val_pred - y_val) ** 2, axis=1)))
                
                history['val_mse'].append(val_mse)
                history['val_mee'].append(val_mee)
                
                # Early stopping
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and epoch % 100 == 0:
                val_info = f", Val MSE: {val_mse:.6f}, Val MEE: {val_mee:.6f}" if X_val is not None else ""
                print(f"Epoch {epoch}: Train MSE: {train_mse:.6f}, Train MEE: {train_mee:.6f}{val_info}")
        
        # Restore best weights if early stopping was used
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
        
        return history
    
    def predict(self, X):
        """
            Make predictions
        """
        activations, _ = self.forward(X)
        return activations[-1]


class KFoldCV:
    """K-Fold Cross Validation"""
    
    @staticmethod
    def split(X, y, k=5, shuffle=True, random_state=None):
        """
            Split data into k folds
            Returns: List of (train_idx, val_idx) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[:n_samples % k] += 1
        
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            folds.append((train_idx, val_idx))
            current = stop
        
        return folds


class GridSearch:
    """Grid Search for hyperparameter tuning"""
    
    def __init__(self, param_grid, k_folds=5, random_state=None):
        """
            Args:
            param_grid: Dictionary with parameter names as keys and lists of values
            k_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.param_grid = param_grid
        self.k_folds = k_folds
        self.random_state = random_state
        self.results = []
    
    def _generate_combinations(self):
        """
            Generate all combinations of parameters
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        def recursive_combine(idx, current):
            if idx == len(keys):
                return [current.copy()]
            
            result = []
            for value in values[idx]:
                current[keys[idx]] = value
                result.extend(recursive_combine(idx + 1, current))
            return result
        
        return recursive_combine(0, {})
    
    def fit(self, X, y, input_size, output_size,
            epochs=1000, batch_size=32,
            early_stopping=50, verbose=True):
        """
            Perform grid search
            Returns: Dictionary with best parameters and all results
        """
        combinations = self._generate_combinations()
        kfold = KFoldCV()
        
        print(f"Testing {len(combinations)} parameter combinations with {self.k_folds}-fold CV...")
        
        for i, params in enumerate(combinations):
            if verbose:
                print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
            
            fold_results = []
            folds = kfold.split(X, y, k=self.k_folds, random_state=self.random_state)
            
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                X_mean_fold = X_train.mean(axis=0)
                X_std_fold = X_train.std(axis=0)

                X_train_norm = (X_train - X_mean_fold)/(X_std_fold + 1e-15)
                X_val_norm = (X_val - X_mean_fold)/(X_std_fold + 1e-15)
                
                # Build layer sizes
                hidden_layers = params.get('hidden_layers', [20])
                layer_sizes = [input_size] + hidden_layers + [output_size]
                
                # Create and train model
                model = NeuralNetwork(
                    layer_sizes=layer_sizes,
                    learning_rate=params.get('learning_rate', 0.01),
                    momentum=params.get('momentum', 0.0),
                    lambda_reg=params.get('lambda_reg', 0.0),
                    activation=params.get('activation', 'tanh')
                )
                
                history = model.train(
                    X_train_norm, y_train, X_val_norm, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    early_stopping_patience=early_stopping,
                    verbose=False
                )
                
                # Get final validation metrics
                val_mse = history['val_mse'][-1] if history['val_mse'] else float('inf')
                val_mee = history['val_mee'][-1] if history['val_mee'] else float('inf')

                # Skip this fold if NaN detected
                if np.isnan(val_mse) or np.isnan(val_mee):
                    print(f"  Fold {fold_idx+1}: NaN detected, skipping")
                    val_mse = float('inf')
                    val_mee = float('inf')
                
                fold_results.append({'mse': val_mse, 'mee': val_mee})
            
            # Average across folds
            avg_mse = np.mean([f['mse'] for f in fold_results])
            std_mse = np.std([f['mse'] for f in fold_results])
            avg_mee = np.mean([f['mee'] for f in fold_results])
            std_mee = np.std([f['mee'] for f in fold_results])
            
            result = {
                'params': params,
                'avg_val_mse': avg_mse,
                'std_val_mse': std_mse,
                'avg_val_mee': avg_mee,
                'std_val_mee': std_mee,
                'fold_results': fold_results
            }
            
            self.results.append(result)
            
            if verbose:
                print(f"  MSE: {avg_mse:.6f} (+/- {std_mse:.6f}), MEE: {avg_mee:.6f} (+/- {std_mee:.6f})")
        
        # Find best parameters (based on MEE as suggested)
        best_idx = np.argmin([r['avg_val_mee'] for r in self.results])
        best_result = self.results[best_idx]
        
        if verbose:
            print(f"\n{'='*30}")
            print(f"Best parameters: {best_result['params']}")
            print(f"Best MEE: {best_result['avg_val_mee']:.6f} (+/- {best_result['std_val_mee']:.6f})")
            print(f"Best MSE: {best_result['avg_val_mse']:.6f} (+/- {best_result['std_val_mse']:.6f})")
            print(f"{'='*30}")
        
        return {
            'best_params': best_result['params'],
            'best_score': best_result['avg_val_mee'],
            'all_results': self.results
        }


# Utility functions
def calculate_mee(y_true, y_pred):
    """ Calculate Mean Euclidean Error """
    return np.mean(np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1)))

def calculate_mse(y_true, y_pred):
    """ Calculate Mean Squared Error """
    return np.mean((y_pred - y_true) ** 2)


def plot_learning_curves(history, save_path='learning_curves.png'):
    """ Plot training and validation learning curves """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_mse']) + 1)
    
    # MSE plot
    axes[0].plot(epochs, history['train_mse'], 'b-', label='Training MSE', linewidth=2)
    if history['val_mse']:
        axes[0].plot(epochs, history['val_mse'], 'r-', label='Validation MSE', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.5)
    
    # MEE plot
    axes[1].plot(epochs, history['train_mee'], 'b-', label='Training MEE', linewidth=2)
    if history['val_mee']:
        axes[1].plot(epochs, history['val_mee'], 'r-', label='Validation MEE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MEE', fontsize=12)
    axes[1].set_title('Mean Euclidean Error', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to '{save_path}'")
    plt.close()
    return

def plot_predictions_vs_true(y_true, y_pred, n_targets, save_path='predictions_vs_true.png'):
    """ Plot scatter plots of predictions vs true values for each target """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_targets):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line (y=x)
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        # Calculate R²
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Calculate MSE for this target
        mse_target = np.mean((y_true[:, i] - y_pred[:, i]) ** 2)
        mee_target = np.mean(np.sqrt((y_true[:, i] - y_pred[:, i])**2))
        
        ax.set_xlabel('True Values', fontsize=11)
        ax.set_ylabel('Predicted Values', fontsize=11)
        ax.set_title(f'Target {i+1} (R²={r2:.4f}, MSE={mse_target:.4f}, MEE={mee_target:.4f})', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction plots saved to '{save_path}'")
    plt.close()
    return



if __name__ == "__main__":
    RANDOM_SEED = 27
    np.random.seed(RANDOM_SEED)

    # EXTRACTING ML-CUP DATASETS
    print("Loading and extracting data.")
    
    train_data = pd.read_csv('CUP_datasets/ML-CUP25-TR.csv', comment='#')
    test_data = pd.read_csv('CUP_datasets/ML-CUP25-TS.csv', comment='#')
    
    # Extract features and targets from training data
    # Known format: ID, feature1, ..., feature12, target1, ..., target4
    train_ids = train_data.iloc[:, 0].values
    X_train_full = train_data.iloc[:, 1:13].values
    y_train_full = train_data.iloc[:, 13:17].values
    
    # Extract features from test data (no targets)
    test_ids = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:13].values
    
    n_features = X_train_full.shape[1]
    n_targets = y_train_full.shape[1]
    
    print(f"Training data: {X_train_full.shape[0]} samples, {n_features} features, {n_targets} targets")
    print(f"Test data: {X_test.shape[0]} samples, {n_features} features")
    print(f"\nFeature statistics:")
    print(f"  X_train mean: {X_train_full.mean():.4f}, std: {X_train_full.std():.4f}")
    print(f"  y_train mean: {y_train_full.mean():.4f}, std: {y_train_full.std():.4f}")
    
    # SPLIT TRAINING DATA
    print("\n"+"="*30)
    print("Splitting training data into development and internal test sets.")
    
    # Split: 80% for development (grid search), 20% for internal test
    n_total = len(X_train_full)
    n_internal_test = int(0.2 * n_total)  # 100 samples for internal test
    
    indices = np.random.permutation(n_total)
    internal_test_idx = indices[:n_internal_test]
    dev_idx = indices[n_internal_test:]
    
    X_dev = X_train_full[dev_idx]
    y_dev = y_train_full[dev_idx]
    X_internal_test = X_train_full[internal_test_idx]
    y_internal_test = y_train_full[internal_test_idx]
    
    print(f"Development set: {len(X_dev)} samples (for grid search)")
    print(f"Internal test set: {len(X_internal_test)} samples (for final evaluation)")
    
    # STANDARDIZING FEATURES
    print("\n"+"="*30)
    print("Features standardization for development set only.")
    
    # Calculate standardization parameters
    X_mean = X_dev.mean(axis=0)
    X_std = X_dev.std(axis=0)
    
    # Apply standardization
    X_dev_norm = (X_dev - X_mean) / (X_std + 1e-8)
    X_internal_test_norm = (X_internal_test - X_mean) / (X_std + 1e-8)
    X_train_full_norm = (X_train_full - X_mean) / (X_std + 1e-8)
    X_test_norm = (X_test - X_mean) / (X_std + 1e-8)
    
    # GRID SEARCH ON DEVELOPMENT SET
    print("\n"+"="*30)
    print("Performing Grid Search with K-Fold Cross-Validation.")
    
    # Defining parameter grid search
    # Coarse
    param_grid = {
        'hidden_layers': [[32], [32, 16], [40, 20]],
        'learning_rate': [1e-6, 1e-5, 1e-4],
        'momentum': [0.3, 0.4, 0.5],
        'lambda_reg': [5e-5, 1e-4, 3e-4],
        'activation': ['tanh']
    }
    
    # Perform grid search on development set
    grid_search = GridSearch(param_grid, k_folds=5, random_state=RANDOM_SEED)
    grid_results = grid_search.fit(
        X_dev, y_dev,
        input_size=n_features,
        output_size=n_targets,
        epochs=1500,
        batch_size=64,
        early_stopping=50,
        verbose=True
    )
    
    best_params = grid_results['best_params']
    
    # RETRAIN ON FULL TRAINING SET
    print("\n"+"="*30)
    print("Retraining model on full training set with best parameters.")
    
    hidden_layers = best_params['hidden_layers']
    layer_sizes = [n_features] + hidden_layers + [n_targets]
    
    final_model = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=best_params['learning_rate'],
        momentum=best_params['momentum'],
        lambda_reg=best_params['lambda_reg'],
        activation=best_params['activation']
    )
    
    # Train on all 500 training samples with internal test as validation for monitoring
    print(f"Training on {len(X_train_full_norm)} samples.")
    print("(Using internal test set for validation monitoring only)")
    history = final_model.train(
        X_dev_norm, y_dev,
        X_val=X_internal_test_norm,
        y_val=y_internal_test,
        epochs=1500,
        batch_size=64,
        early_stopping_patience=50,
        verbose=True
    )
    
    # PLOT LEARNING CURVES
    print("\n"+"="*30)
    print("Plotting learning curves.")
    plot_learning_curves(history, save_path='learning_curves.png')

    # EVALUATING BEST MODEL ON INTERNAL TEST SET
    print("\n"+"="*30)
    print("Evaluating on Internal Test Set.")
    
    # Predictions on internal test set
    predictions_internal = final_model.predict(X_internal_test_norm)
    
    # Calculate metrics
    internal_mse = calculate_mse(y_internal_test, predictions_internal)
    internal_mee = calculate_mee(y_internal_test, predictions_internal)
    
    print("\nModel performance on internal test set.")
    print(f"    MSE: {internal_mse:.6f}")
    print(f"    MEE: {internal_mee:.6f}")
    
    # Per-target MSE
    print("\nPer-target MSE:")
    for i in range(n_targets):
        target_mse = np.mean((predictions_internal[:, i] - y_internal_test[:, i]) ** 2)
        print(f"  Target {i+1}: {target_mse:.6f}")

    print("\nPer-target MEE:")
    for i in range(n_targets):
        target_mee = np.mean(np.sqrt((predictions_internal[:, i] - y_internal_test[:, i]) ** 2))
        print(f"  Target {i+1}: {target_mee:.6f}")
    
    # PLOT PREDICTIONS VS TRUE VALUES
    print("\n"+"="*30)
    print("Plotting predictions vs true values over the retraining.")
    plot_predictions_vs_true(y_internal_test, predictions_internal, n_targets, 
                            save_path='predictions_vs_true.png')
    
    # PREDICT ON FINAL TEST SET
    print("\n"+"="*30)
    print("Making predictions on the final test set.")
    
    predictions_test = final_model.predict(X_test_norm)
    
    print(f"Generated {len(predictions_test)} predictions.")
    
    # SAVING RESULTS
    print("\n"+"="*30)
    print("Saving results:")
    print(" MaGiC_ML-CUP25-TS.csv contains the predictions over the test set")
    print(" model_results.json contains the parameters resulting from the Search Grid")
    
    # Save predictions
    output_file = 'MaGiC_ML-CUP25-TS.csv'

    # Write header comments
    with open(output_file, "w") as f:
        f.write("# Battisti Matilde, Tarasi Luigi\n")
        f.write("# MaGiC\n")
        f.write("# ML-CUP25 v1\n")
        f.write(f"# {datetime.now().strftime('%d/%m/%Y')}\n")

    results_df = pd.DataFrame(
        predictions_test,
        columns=[f'target_{i+1}' for i in range(n_targets)]
    )
    results_df.insert(0, 'ID', test_ids)

    results_df.to_csv(output_file, mode='a', index=False, header=False)
    print(f"Predictions saved to {output_file}")
    
    # Save detailed results
    detailed_results = {
        'best_parameters': best_params,
        'batch_size': '64',
        'cv_best_mee': float(grid_results['best_score']),
        'internal_test_mse': float(internal_mse),
        'internal_test_mee': float(internal_mee),
        'architecture': layer_sizes,
        'training_samples': len(X_train_full),
        'internal_test_samples': len(X_internal_test)
    }
    
    with open('model_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print("Detailed results saved to 'model_results.json'")
    
    # Print top 10 configurations from grid search
    all_results = grid_results['all_results']
    sorted_results = sorted(all_results, key=lambda x: x['avg_val_mee'])
    
    print("\nTop 10 configurations from grid search.")
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"\n{i}. MEE: {result['avg_val_mee']:.6f} (+/- {result['std_val_mee']:.6f})")
        print(f"   Params: {result['params']}")

    