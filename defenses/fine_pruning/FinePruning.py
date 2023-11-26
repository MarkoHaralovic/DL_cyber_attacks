import sys
import os
import torch
import torch.nn as nn


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)


from notebooks.Data import Data


train_images = "..\\..\\datasets\\CIFAR10\\cifar-10\\train\\data.npy"
train_labels = "..\\..\\datasets\\CIFAR10\\cifar-10\\train\\labels.npy"
test_images = "..\\..\\datasets\\CIFAR10\\cifar-10\\test\\data.npy"
test_labels = "..\\..\\datasets\\CIFAR10\\cifar-10\\test\\labels.npy"
weight_path = "..\\..\\models\\efficientnet_v2_s_cifar10.pth"
model_name = 'efficientnet_v2_s'

cifar_10_dataset= Data(train_images=train_images,train_labels=train_labels,
                     test_images=test_images,test_labels=test_labels)

cifar_10_dataset.show_images()


"""
This defense is based on two types of defenses:
      - fine tuning
      - pruning
The pruning defense works as follows: the defender exercises the DNN received from the attacker with clean inputs from the validation dataset, Dvalid, 
and records the average activation of each neuron. The defender then iteratively prunes neurons from the DNN in increasing order of average activations
and records the accuracy of the pruned network in each iteration. The defense terminates when the accuracy on the validation dataset drops below a
pre-determined threshold.

Pruning has three phases :
   Phase One : the neurons in the first phase are not activated by either clean or backdoored inputs
   Phase Two : the neurons in the second phase are activated by backdoored inputs
   Phase Three : the neurons in the third phase are activated by clean inputs
    
Fine-tuning uses the pre-trained DNN weights to initialize training (instead of random initialization) and a smaller learning rate since the final weights
are expected to be relatively close to the pretrained weights. Fine-tuning is significantly faster than training a network from scratch

Fine Pruning:  first prunes the DNN returned by the attacker and then fine-tunes the pruned network.
"""

class FinePruning():
   def __init__ (self):
      pass
   
