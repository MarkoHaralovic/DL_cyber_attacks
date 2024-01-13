If there are already preapred models the first two steps in SD can be skipped.
The current version expects .tar files.
The paths currently expect running all the steps, should be changed to test the attacks.
The paths also currently save to the current directory and create subdirectories and the paths for models are in the subdirectories, so paths should be changed.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Sample-Distinguishment (SD) Module

Step1: Train a backdoored model without any data augmentations.
python train_attack_noTrans.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 2

Step2: Fine-tune the backdoored model with intra-class loss.
python finetune_attack_noTrans.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 10 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans/cifar10/resnet18/gridTrigger/1.tar

Step3: Calculate the values of the FCT metric for all training samples.
python calculate_consistency.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans_ftsimi/cifar10/resnet18/gridTrigger/9.tar

If you want to visualize values of the FCT metric, you can run:
python visualize_consistency.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans_ftsimi/cifar10/resnet18/gridTrigger/9.tar

Step4: Calculate thresholds for choosing clean and poisoned samples.
python calculate_gamma.py --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans_ftsimi/cifar10/resnet18/gridTrigger/9.tar 

Step5: Separate training samples into clean samples, poisoned samples and uncertain samples.
python separate_samples.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --batch_size 1 --clean_ratio 0.20 --poison_ratio 0.05 --gamma_low x.x --gamma_high y.y --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/noTrans_ftsimi/cifar10/resnet18/gridTrigger/9.tar

gamma_low and gamma_high in Step5 are obtained in Step4, so x.x and y.y are supposed to be replaced with the obtained values.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
two-stage Secure Training (ST) Module

cd ST

Step1: Train the feature extractor via semi-supervised contrastive learning.
python train_extractor.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 200 --learning_rate 0.5 --temp 0.1 --cosine --save_freq 20 --batch_size 512

Parameters are set as the same in [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf) (https://github.com/HobbitLong/SupContrast).

Step2: Train the classifier via minimizing a mixed cross-entropy loss.
python train_classifier.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 10 --learning_rate 5 --batch_size 512 --ckpt ./save/poison_rate_0.1/SupCon_models/cifar10/resnet18/gridTrigger_0.2_0.05/SupCon_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth

Parameters are set as the same in Supervised Contrastive Learning.

Step3: Test the final model.

python test.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --model_ckpt ./save/poison_rate_0.1/SupCon_models/cifar10/resnet18/gridTrigger_0.2_0.05/SupCon_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth --classifier_ckpt ./save/poison_rate_0.1/SupCon_models/cifar10/resnet18/gridTrigger_0.2_0.05/Linear_cifar10_resnet18_lr_5.0_decay_0_bsz_512/ckpt_epoch_9.pth

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Backdoor Removal (BR) Module

Step1: Train a backdoored model with classical data augmentations.

python train_attack_withTrans.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 200

Step2: Unlearn and relearn the backdoored model.
python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type gridTrigger --epochs 20 --clean_ratio 0.20 --poison_ratio 0.05 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/gridTrigger/199.tar --checkpoint_save ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/gridTrigger/199_unlearn_purify.py --log ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/gridTrigger/unlearn_purify.csv

