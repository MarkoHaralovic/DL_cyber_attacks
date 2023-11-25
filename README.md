# Backdoor Attacks and Defenses  on Deep Neural Networks

This README provides an overview of various backdoor attacks and defenses on deep neural networks, summarizing their key properties and additional notes where relevant.

## Overview

Backdoor attacks in deep learning are a form of adversarial attack where a model is manipulated to respond to certain trigger inputs in a predetermined way, often while performing normally on standard inputs. 

Similarily, backdoor defense in Deep Neural Networks is a methodology to evade such attacks.


## Backdoor Defenses

| Method            | Article Name                                                                                                   | Link |
|-------------------|----------------------------------------------------------------------------------------------------------------|------|
| Neural Cleanse    | Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Network | [#](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf) |
| Fine-Pruning           | Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks  | [#](https://www.researchgate.net/profile/Siddharth-Garg/publication/325483856_Fine-Pruning_[…]ng-Against-Backdooring-Attacks-on-Deep-Neural-Networks.pdf) |
| Jittering         | Effective Backdoor Defense by Exploiting Sensitivity of Poisoned Samples  | [#](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3f9bbf77fbd858e5b6e39d39fe84ed2e-Abstract-Conference.html) |

## Backdoor Attacks

| Method     | Article Name                                                                       | Link |
|------------|------------------------------------------------------------------------------------|------|
| BadNets    | BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain   | [#](https://arxiv.org/abs/1708.06733) |
| Data Poisoning  | Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning                   | [#](https://arxiv.org/abs/1712.05526) |


## Dataset

All experiments will be conducted on CIFAR-10 dataset.
The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

10 classes present in the dataset are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

