# Backdoor Attacks and Defenses  on Deep Neural Networks

This README provides an overview of various backdoor attacks and defenses on deep neural networks, summarizing their key properties and additional notes where relevant.

## Overview

Backdoor attacks in deep learning are a form of adversarial attack where a model is manipulated to respond to certain trigger inputs in a predetermined way, often while performing normally on standard inputs. 

Similarily, backdoor defense in Deep Neural Networks is a methodology to evade such attacks.


## Backdoor Defenses

| Method            | Article Name                                                                                                   | Link |
|-------------------|----------------------------------------------------------------------------------------------------------------|------|
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

## Setup

All dependencies are provided in requirements.txt. Run command pip install -r requirements.txt 
to install requirements to run this project locally

## Running Experiments

### Model training

To train Resnet 18 model, position yourself in the /notebooks folder. In resnet18_train.py set CLEAN to True to train on clean dataset, or to False to train on poisoned dataset, and if doing later, please set the paths to the /datasets folder containing desired poisoned training data. 

Start training Resnet with command:

```bash
python resnet18_train.py
```

To train Efficient Net B0 model, position yourself in the /notebooks folder and repeat process as for Resnet training in notebook efficientnet_v2_cifar10.py, and run command

```bash
python efficientnet_v2_cifar10.py
```

### Fine Pruning

After creating the poisoned dataset, with either one of the attacks (data poisoning or bad nets), position yourself in folder DL_cyber_attacks\defenses\fine_pruning\efficientNet to fine-tune EfficientNet or in DL_cyber_attacks\defenses\fine_pruning\resnet to fine-prune Resnet 18 model.

In config.json, modify the  weight path to the location of the weights file of a model you want to prune and tune. Define the pruning rates, or leave as is to conduct experiment for all rates. Specify which layer to prune in 'layer_keys', or leave as is to incrementally prune all sub-layers in the last layer. Define the batch size for testing, based on your memory availability, and set the learning rate for the fine tuning.

Then, run the following command :

```bash
python Pruning.py
```

In this process, only csv records will be updated. These contain information about the layer, prune rates, accuracy, and both targeted and untargeted attack success rates.

Based on the results of the pruning, decide for a prune rate and the layer to prune, modify those parameters in config.json and run the following command:

```bash
python FinePruning.py
```

Pruned and tuned model will be saved on location DL_cyber_attacks\models.

### Jittering

Position yourself into the /defenses/jittering folder.

------------------------------------------------------------------------------------------------------------------
Sample-Distinguishment (SD) Module

#### Step1: Train a backdoored model without any data augmentations.

```bash
python train_attack_noTrans.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 2
```

#### Step2: Fine-tune the backdoored model with intra-class loss.

```bash
python finetune_attack_noTrans.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans/cifar10/resnet18/gridTrigger/1.tar
```

#### Step3: Calculate the values of the FCT metric for all training samples.

```bash
python calculate_consistency.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans_ftsimi/cifar10/resnet18/gridTrigger/9.tar
```

If you want to visualize values of the FCT metric, you can run:
```bash
python visualize_consistency.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans_ftsimi/cifar10/resnet18/gridTrigger/9.tar
```

#### Step4: Calculate thresholds for choosing clean and poisoned samples.

```bash
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans_ftsimi/cifar10/resnet18/gridTrigger/9.tar 
```

#### Step5: Separate training samples into clean samples, poisoned samples and uncertain samples.

```bash
python separate_samples.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low x.x --gamma_high y.y --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans_ftsimi/cifar10/resnet18/gridTrigger/9.tar
```

gamma_low and gamma_high in Step5 are obtained in Step4, so x.x and y.y are supposed to be replaced with the obtained values.

-------------------------------------------------------------------------------------------------------------------------------

### two-stage Secure Training (ST) Module
```bash
cd ST
```

#### Step1: Train the feature extractor via semi-supervised contrastive learning.

```bash
python train_extractor.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512
```

Parameters are set as the same in [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf) (https://github.com/HobbitLong/SupContrast).

#### Step2: Train the classifier via minimizing a mixed cross-entropy loss.

```bash
python train_classifier.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.1/SupCon_models/cifar10/resnet18/gridTrigger_0.2_0.05/SupCon_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth
```

Parameters are set as the same in Supervised Contrastive Learning.

#### Step3: Test the final model.

```bash
python test.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --model_ckpt ./save/poison_rate_0.1/SupCon_models/cifar10/resnet18/gridTrigger_0.2_0.05/SupCon_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth --classifier_ckpt ./save/poison_rate_0.1/SupCon_models/cifar10/resnet18/gridTrigger_0.2_0.05/Linear_cifar10_resnet18_lr_5.0_decay_0_bsz_512/ckpt_epoch_9.pth
```

-------------------------------------------------------------------------------------------------------------------------------
### Backdoor Removal (BR) Module

#### Step1: Train a backdoored model with classical data augmentations.

```bash
python train_attack_withTrans.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 200
```

#### Step2: Unlearn and relearn the backdoored model.

```bash
python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 20 --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/gridTrigger/199.tar --checkpoint_save ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/gridTrigger/199_unlearn_purify.py --log ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/gridTrigger/unlearn_purify.csv
```
