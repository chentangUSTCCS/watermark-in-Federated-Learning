# Watermark in Federated Learning (PyTorch)

Watermark in Federated Learning.


Experiments are produced on CIFAR-10 and CIFAR-100 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

Network are AlexNet and ResNet18.

The code references the following two repositories：
https://github.com/kamwoh/DeepIPR & https://github.com/AshwinRJ/Federated-Learning-PyTorch.

## Requirements
Install all the packages
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on  CIFAR-10 and CIFAR-100.
* To use your own dataset: Move your dataset to data directory `data/` and write a wrapper on pytorch dataset class.

## Running the experiments

- To run the capacity analysis in section 3 when embeeding watermakr into different positions:

```
python FL_capacity_alexnet.py --epochs=100 --num_users=5 --frac=1 --dataset=cifar10 --model=alexnet --cuda_id=3 --iid=1 --W=400 --b=256 --coe=1
python FL_capacity_resnet.py --epochs=200 --num_users=5 --frac=1 --dataset=cifar100 --model=resnet18 --cuda_id=3 --iid=1 --W=400 --b=256 --local_bs=16 --local_ep=2 --lr=0.01
```

- To run the capacity analysis in section 3 when embeeding watermark into same positions:

```
python FL_capacity_alexnet1.py --epochs=100 --num_users=7 --frac=1 --dataset=cifar10 --model=alexnet --cuda_id=3 --iid=1 --W=2000 --b=256 --coe=1
python FL_capacity_resnet1.py --epochs=200 --num_users=29 --frac=1 --dataset=cifar100 --model=resnet18 --cuda_id=3 --iid=1 --W=30976 --b=256 --local_bs=16 --local_ep=2 --lr=0.01 --coe=1
```

- To run the baseline without watermark embedding:

```
python FL_baseline.py --epochs=200 --num_users=50 --frac=1 --dataset=cifar100 --model=inception --cuda_id=3 --iid=1
```
- To run the watermark embedding under IID condition:

```
alexnet:
python FL_BloomFilter.py --epochs=200 --num_users=45 --frac=1 --coe=0.1 --dataset=cifar10 --model=alexnet --cuda_id=0 --iid=1 --position=features.5.conv.weight --local_ep=2
resnet18:
python FL_BloomFilter.py --epochs=200 --num_users=50 --frac=1 --coe=0.1 --dataset=cifar100 --model=resnet18 --cuda_id=3 --iid=1 --local_ep=2 --position=layer4.0.convbn_2.conv.weight --local_bs=16 --lr=0.01
```
## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'cifar10'. Options: 'cifar100'
* ```--model:```     Default: 'alexnet'. Options: 'vgg', 'resnet18'
* ```--cuda_id:```  Default: '0' (runs on GPU).
* ```--epochs:```   Number of rounds of training.
* ```--lr:```         Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```      Random Seed. Default set to 1.

#### Federated Parameters
* `--W:` Length of parameters to embed watermark.
* `--X:` Length of embedding parameters.
* `--b:` Length of watermark.
* `--H:` Number of hash functions.
* ```--iid:``` Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:``` Number of users. Default is 20.
* ```--frac:```  Fraction of users to be used for federated updates. Default is 1.
* `--coe:` watermark coefficient of loss function.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.

## contact us
mail: chentang1999@mail.ustc.edu.cn
