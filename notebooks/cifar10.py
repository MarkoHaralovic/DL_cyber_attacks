import os
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_batch(batch):
    """
    CIFAR-10 dataset description:
    ->data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values,
              the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the
              red channel values of the first row of the image.
    ->labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    
    """
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)  # reshape and transpose
    
    return data, labels

cifar_10_dataset_dir = "datasets/CIFAR10/cifar-10-batches-py"
train_output_dir = "datasets/CIFAR10/cifar-10/train"
test_output_dir = "datasets/CIFAR10/cifar-10/test"

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)


num_train_samples = 50000  # CIFAR-10 has 50,000 training samples
x_train = np.empty((num_train_samples, 32, 32, 3), dtype="uint8")
y_train = np.empty((num_train_samples,), dtype="uint8")

for i in range(1, 6):
    file_path = os.path.join(cifar_10_dataset_dir, "data_batch_" + str(i))
    batch = unpickle(file_path)
    start_idx = (i - 1) * 10000
    end_idx = i * 10000
    x_train[start_idx:end_idx, :, :, :], y_train[start_idx:end_idx] = load_cifar10_batch(batch)
    
np.save(os.path.join(train_output_dir, "data.npy"), x_train)
np.save(os.path.join(train_output_dir, "labels.npy"), y_train)

file_path = os.path.join(cifar_10_dataset_dir, "test_batch")
test_batch = unpickle(file_path)
x_test, y_test = load_cifar10_batch(test_batch)
x_test = np.array(x_test, dtype='uint8')
y_test = np.array(y_test, dtype='uint8')

np.save(os.path.join(test_output_dir, "data.npy"), x_test)
np.save(os.path.join(test_output_dir, "labels.npy"), y_test)

print('Total number of Images in the Dataset:', len(x_train) + len(x_test))
print('Number of train images:', len(x_train))
print('Number of test images:', len(x_test))
print('Shape of training dataset:',x_train.shape)
print('Shape of testing dataset:',x_test.shape)