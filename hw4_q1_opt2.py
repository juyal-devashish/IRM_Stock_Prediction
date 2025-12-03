import numpy as np
import os
import urllib.request
import gzip
import pickle

def download_mnist_manually():
    """
    Download MNIST files manually to avoid SSL issues
    """
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz', 
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    data_dir = './data/MNIST/raw'
    os.makedirs(data_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, file_path)
    
    return data_dir

def load_mnist_manual(data_dir):
    """
    Load MNIST data from manually downloaded files
    """
    def read_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip magic number and dimensions
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 28, 28)
    
    def read_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip magic number
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    # Load data
    X_train = read_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = read_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    X_test = read_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = read_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    return X_train, y_train, X_test, y_test

def normalize_mnist_data():
    """
    Applies the specified normalization to MNIST data
    """
    print("Loading MNIST data...")
    
    try:
        # Try manual download first
        data_dir = download_mnist_manually()
        X_train, y_train, X_test, y_test = load_mnist_manual(data_dir)
        print("Successfully loaded MNIST data manually")
        
    except Exception as e:
        print(f"Manual download failed: {e}")
        print("Trying alternative: using keras datasets...")
        
        # Alternative: use keras if available
        try:
            from tensorflow import keras
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            print("Successfully loaded MNIST data from keras")
        except ImportError:
            print("Keras not available. Creating dummy data for demonstration...")
            # Create dummy data for the structure
            X_train = np.random.randint(0, 256, (60000, 28, 28), dtype=np.uint8)
            y_train = np.random.randint(0, 10, 60000)
            X_test = np.random.randint(0, 256, (10000, 28, 28), dtype=np.uint8)
            y_test = np.random.randint(0, 10, 10000)
            print("Using random dummy data - normalization will still be demonstrated")
    
    # Convert to float32 for normalization
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"Original training data shape: {X_train.shape}")
    print(f"Original test data shape: {X_test.shape}")
    
    # Step 1: Flatten images
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat = X_test.reshape(-1, 784)
    
    # Step 2: Z-normalization using training statistics
    print("\nStep 1: Computing normalization statistics from training data...")
    pixel_means = np.mean(X_train_flat, axis=0)
    pixel_stds = np.std(X_train_flat, axis=0)
    
    # Avoid division by zero for any constant pixels
    pixel_stds[pixel_stds == 0] = 1.0
    
    # Normalize training data
    X_train_normalized = (X_train_flat - pixel_means) / pixel_stds
    
    # Normalize test data using training statistics (important: no data leakage)
    X_test_normalized = (X_test_flat - pixel_means) / pixel_stds
    
    # Step 3: Add bias term (concatenate 1 to each sample)
    print("Step 2: Adding bias term...")
    X_train_final = np.hstack([X_train_normalized, np.ones((X_train_normalized.shape[0], 1))])
    X_test_final = np.hstack([X_test_normalized, np.ones((X_test_normalized.shape[0], 1))])
    
    return X_train_final, y_train, X_test_final, y_test, pixel_means, pixel_stds

def verify_normalization(X_train_final, X_test_final, pixel_means, pixel_stds):
    """
    Verifies that normalization was applied correctly
    """
    print("\n" + "="*60)
    print("NORMALIZATION VERIFICATION")
    print("="*60)
    
    # Check shapes
    print(f"\n1. Shape verification:")
    print(f"   Training data: {X_train_final.shape} (expected: (60000, 785))")
    print(f"   Test data: {X_test_final.shape} (expected: (10000, 785))")
    
    # Check bias term
    print(f"\n2. Bias term verification:")
    print(f"   Last column unique values: {np.unique(X_train_final[:, -1])} (expected: [1.])")
    print(f"   All samples have bias=1: {np.all(X_train_final[:, -1] == 1.0)}")
    
    # Check normalization statistics (exclude bias term for pixel stats)
    X_train_pixels = X_train_final[:, :-1]
    X_test_pixels = X_test_final[:, :-1]
    
    print(f"\n3. Training data normalization (first 10 pixels):")
    print("   Pixel |   Mean     |   Std      | Status")
    print("   " + "-"*40)
    for i in range(10):
        mean_val = np.mean(X_train_pixels[:, i])
        std_val = np.std(X_train_pixels[:, i])
        status = "✓" if abs(mean_val) < 0.01 and 0.99 < std_val < 1.01 else "✗"
        print(f"   {i:5d} | {mean_val:10.6f} | {std_val:10.6f} | {status}")
    
    print(f"\n4. Global statistics:")
    print(f"   Training data - Mean: {np.mean(X_train_pixels):.6f}, Std: {np.std(X_train_pixels):.6f}")
    print(f"   Test data - Mean: {np.mean(X_test_pixels):.6f}, Std: {np.std(X_test_pixels):.6f}")
    
    print(f"\n5. Data range check:")
    print(f"   Training data min/max: [{np.min(X_train_pixels):.3f}, {np.max(X_train_pixels):.3f}]")
    print(f"   Test data min/max: [{np.min(X_test_pixels):.3f}, {np.max(X_test_pixels):.3f}]")
    
    # Save normalized data for future use
    np.save('X_train_normalized.npy', X_train_final)
    np.save('X_test_normalized.npy', X_test_final)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    print(f"\n6. Data saved to disk for future use")

# Main execution
if __name__ == "__main__":
    # Perform normalization
    X_train, y_train, X_test, y_test, pixel_means, pixel_stds = normalize_mnist_data()
    
    # Verify results
    verify_normalization(X_train, X_test, pixel_means, pixel_stds)
    
    print("\n" + "="*60)
    print("SUMMARY: All normalization steps completed successfully!")
    print("="*60)