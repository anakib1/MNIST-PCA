import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def load_mnist():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = y.astype(np.int64)
    return X, y

def create_pixel_mask(X, percentile):
    """Create a mask of informative pixels using PCA."""
    print(f"Performing PCA analysis with {percentile}% pixels kept...")
    pca = PCA(n_components=0.95)
    pca.fit(X)
    
    # Calculate the importance of each pixel
    pixel_importance = np.abs(pca.components_).sum(axis=0)
    
    # Create a mask for pixels that contribute to the selected components
    mask = pixel_importance > np.percentile(pixel_importance, 100 - percentile)
    
    print(f"Number of pixels kept: {mask.sum()}")
    print(f"Number of pixels removed: {len(mask) - mask.sum()}")
    
    return mask, pixel_importance

def visualize_mask(mask, pixel_importance, percentile, example_image, save_dir='masks'):
    """Visualize the mask and pixel importance."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(15, 5))
    
    # Plot the mask
    plt.subplot(1, 3, 1)
    plt.imshow(mask.reshape(28, 28), cmap='gray')
    plt.title(f'Pixel Mask ({percentile}% kept)')
    
    # Plot pixel importance
    plt.subplot(1, 3, 2)
    plt.imshow(pixel_importance.reshape(28, 28), cmap='hot')
    plt.title('Pixel Importance')
    plt.colorbar()
    
    # Plot example masked image
    plt.subplot(1, 3, 3)
    masked_image = example_image.reshape(28, 28) * mask.reshape(28, 28)
    plt.imshow(masked_image, cmap='gray')
    plt.title('Example Masked Image')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/mask_{percentile}percent.png')
    plt.close()

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(model, train_loader, val_loader, epochs=5):
    """Train the neural network model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

def main():
    # Load and prepare data
    X, y = load_mnist()
    
    # Split data into train and validation sets
    train_size = 50000
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Select an example image for visualization
    example_image = X_train[0]  # Using the first training image
    
    # Test different mask sizes
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accuracies = []
    pixel_counts = []
    
    for percentile in percentiles:
        # Create pixel mask using PCA
        mask, pixel_importance = create_pixel_mask(X_train, percentile)
        
        # Visualize the mask
        visualize_mask(mask, pixel_importance, percentile, example_image)
        
        # Apply mask to data
        X_train_masked = X_train[:, mask]
        X_val_masked = X_val[:, mask]
        
        # Convert to PyTorch tensors
        X_train_masked_tensor = torch.FloatTensor(X_train_masked)
        X_val_masked_tensor = torch.FloatTensor(X_val_masked)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create data loaders
        train_masked_dataset = TensorDataset(X_train_masked_tensor, y_train_tensor)
        val_masked_dataset = TensorDataset(X_val_masked_tensor, y_val_tensor)
        train_masked_loader = DataLoader(train_masked_dataset, batch_size=64, shuffle=True)
        val_masked_loader = DataLoader(val_masked_dataset, batch_size=64)
        
        # Train model
        print(f"\nTraining model with {percentile}% of pixels...")
        masked_model = SimpleNN(mask.sum())
        accuracy = train_model(masked_model, train_masked_loader, val_masked_loader)
        
        accuracies.append(accuracy)
        pixel_counts.append(mask.sum())
        
        print(f"Best validation accuracy: {accuracy:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy vs number of pixels
    plt.subplot(1, 2, 1)
    plt.plot(pixel_counts, accuracies, 'bo-')
    plt.xlabel('Number of Pixels')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Performance vs Number of Pixels')
    plt.grid(True)
    
    # Plot accuracy vs percentage of pixels kept
    plt.subplot(1, 2, 2)
    plt.plot(percentiles, accuracies, 'ro-')
    plt.xlabel('Percentage of Pixels Kept')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Performance vs Percentage of Pixels')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    plt.close()
    
    # Save results to file
    with open('performance_results.txt', 'w') as f:
        f.write("Performance Analysis Results\n")
        f.write("==========================\n\n")
        for i, (p, acc, pix) in enumerate(zip(percentiles, accuracies, pixel_counts)):
            f.write(f"Test {i+1}:\n")
            f.write(f"Percentage of pixels kept: {p}%\n")
            f.write(f"Number of pixels: {pix}\n")
            f.write(f"Validation accuracy: {acc:.2f}%\n\n")

if __name__ == "__main__":
    main() 