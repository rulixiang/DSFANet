# DSFANet (Deep Slow Feature Analysis Net)
Implementioan of [Unsupervised Deep Slow Feature Analysis for Change Detection in Multi-Temporal Remote Sensing Images](https://arxiv.org/abs/1812.00645)

<img src="./figures/dsfa.png">

## Requirements

```
tensorflow_gpu==1.7.0
scipy==1.0.0
numpy==1.14.0
matplotlib==2.1.2
tensorflow==1.14.0
```

## Usage
```
pip install -r requirements.txt
```
```
usage: python dsfa.py [-h] [-e EPOCH] [-l LR] [-r REG] [-t TRN] [-i ITER] [-g GPU]
               [--area AREA]
```
```
optional arguments:
  -h, --help              show this help message and exit
  -e EPOCH, --epoch EPOCH epoches
  -l LR, --lr LR          learning rate
  -r REG, --reg REG       regurlization parameter
  -t TRN, --trn TRN       number of training samples
  -i ITER, --iter ITER    max iteration
  -g GPU, --gpu GPU       GPU ID
  --area AREA             datasets

```

## Citation
Please cite our paper if you use this code in your research.
```
@article{du2018unsupervised,
  title={Unsupervised Deep Slow Feature Analysis for Change Detection in Multi-Temporal Remote Sensing Images},
  author={Du, Bo and Ru, Lixiang and Wu, Chen and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2019}
}
```
The data in this repo is originally provided in [GETNET: A General End-to-end Two-dimensional CNN Framework for Hyperspectral Image Change Detection.](https://arxiv.org/abs/1905.01662)

## Q & A
**For any questions, please [contact me.](mailto:rulixiang@outlook.com)**
