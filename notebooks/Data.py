import numpy as np
from torchvision import transforms 
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Data:
   def __init__(self, train_images,train_labels, test_images, test_labels, train_transform = None,test_transform = None):
      """
      Expecting 3 paths to train and test npy arrays (both for labels and images)
      Optional: transforms, if sent, implementation is needed, currently Data class
      contains helper fucntions ot achieve dataset transformations.
      """
      self.train_images = np.load(train_images)
      self.train_labels = np.load(train_labels)
      self.test_images= np.load(test_images)
      self.test_labels = np.load(test_labels)
      self.shape = self.test_images[0].shape
      self.train_transform = train_transform
      self.test_transform = test_transform
      self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
         ]
      self.num_classes = len(self.classes)

      
   
   def __len__(self):
      return len(self.train_images) + len(self.test_images)
   

   def __getitem__(self, idx, dataset = "train"):
      """
      Generic get item function to get one image from train dataset
      """
      sample, label = self.train_images[idx], self.train_labels[idx]
      if self.train_transform:
         sample = self.train_transform(sample)
      return sample, label
   def visualize_class_images(self, class_name, n):
    """
    Visualize n instances of a specific class of the dataset.
    """

    class_idx = self.classes.index(class_name)
    
    indices = np.where(self.train_labels == class_idx)[0]

    selected_indices = np.random.choice(indices, n, replace=False)
    if n == 1:
        fig, axes = plt.subplots(figsize=(3, 3))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    
    for i, idx in enumerate(selected_indices):
        img = self.train_images[idx]
        img = img.transpose(0,1,2)
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.show()

      
   def normalize(self):
      """
      Data normalization
      """
      self.train_images = self.train_images.astype("float32")
      self.train_images = self.train_images / 255.0
      self.test_images = self.test_images.astype("float32")
      self.test_images = self.test_images / 255.0
      
      return
   def reshape(self):
      """
      Reshaping image dimensions ( default is (32,32,3) )
      """
      return
      
   def resize(self, resized_shape):
      """
      Image resize, whereas each image in dataset has shape (32,32,3),
      but works for permuted dimensions as well (3,32,32)
      """ 
      
      resize_transform = transforms.Resize(resized_shape, antialias=True)
      
      if self.shape == (32, 32, 3):
            self.train_images = np.array([np.array(resize_transform(Image.fromarray((img * 255).astype('uint8')))) for img in tqdm(self.train_images, desc="Resizing train images")])
            self.test_images = np.array([np.array(resize_transform(Image.fromarray((img * 255).astype('uint8')))) for img in tqdm(self.test_images, desc="Resizing test images")])
      else:
         self.train_images = np.array([np.array(resize_transform(Image.fromarray(img.transpose(1, 2, 0) * 255).astype('uint8'))) for img in tqdm(self.train_images, desc="Resizing train images")]).transpose(0, 3, 1, 2)
         self.test_images = np.array([np.array(resize_transform(Image.fromarray(img.transpose(1, 2, 0) * 255).astype('uint8'))) for img in tqdm(self.test_images, desc="Resizing test images")]).transpose(0, 3, 1, 2)

      print("Resizing completed")
      return
      
      
   def one_hot_encoding(self):
      """   
      One-hot encoding
      Represent each integer value as a binary vector that is all zeros except the index of the integer
      """
      if self.train_images:
         self.train_hot = np_utils.np_utils.to_categorical(self.train_images)
      if self.test_images:
         self.test_hot = np_utils.np_utils.to_categorical(self.test_images)
      if self.valid_images:
         self.valid_hot = np_utils.np_utils.to_categorical(self.valid_images)
      
      return self.train_hot,self.test_hot, self.valid_hot
               
   def mean_std(self, dataset):
      """
      Calculates mean and standard deviation across each channel
      """
      if dataset == "train":
         mean_r = self.train_images[:,:,:,0].mean()
         mean_g = self.train_images[:,:,:,1].mean()
         mean_b = self.train_images[:,:,:,2].mean()

         std_r = self.train_images[:,:,:,0].std()
         std_g = self.train_images[:,:,:,1].std()
         std_b = self.train_images[:,:,:,2].std()
      elif dataset == "test":
         mean_r = self.test_images[:,:,:,0].mean()
         mean_g = self.test_images[:,:,:,1].mean()
         mean_b = self.test_images[:,:,:,2].mean()

         std_r = self.test_images[:,:,:,0].std()
         std_g = self.test_images[:,:,:,1].std()
         std_b = self.test_images[:,:,:,2].std()
      elif dataset == "valid":
         mean_r = self.valid_images[:,:,:,0].mean()
         mean_g = self.valid_images[:,:,:,1].mean()
         mean_b = self.valid_images[:,:,:,2].mean()

         std_r = self.valid_images[:,:,:,0].std()
         std_g = self.valid_images[:,:,:,1].std()
         std_b = self.valid_images[:,:,:,2].std()
      else:
         raise ValueError("Invalid dataset type: %s" % dataset)

      return mean_r,mean_g,mean_b,std_r, std_g, std_b
   
   def create_valid_images(self, ratio = 0.25):
      """"
      Ratio is percent of train imaes that will be used as validation dataset.
      Default value is 25%  (sckit learn documentation)
      """
      train_images, valid_images, train_labels, valid_labels = train_test_split(self.train_images, self.train_labels, test_size=0.25)
      self.train_images = train_images
      self.train_labels = train_labels
      self.valid_images  = valid_images
      self.valid_labels = valid_labels
      
      return
   
   def show_images(self):
      """
      Vizualizing one image per class 
      
      """
      num_train, img_channels, img_rows, img_cols = self.train_images.shape
      num_classes = len(np.unique(self.classes))
      fig = plt.figure(figsize=(10, 3))
      
      for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(self.train_labels[:] == i)[0]
        x_idx = self.train_images[idx, ::]
        img_num = np.random.randint(x_idx.shape[0])
        im = np.transpose(x_idx[img_num, ::], (0, 1, 2))
        ax.set_title(self.classes[i])
        plt.imshow(im)

      plt.show()
      
      
      
      