# FedVAD: Enhancing Federated Video Anomaly Detection with GPT-Driven Semantic Distillation
## Introduction
This repo is the official implementation of "FedVAD: Enhancing Federated Video Anomaly Detection with GPT-Driven Semantic Distillation" (under review). 
## Requirements
The code requires ```python>=3.8``` and the following packages:
```
torch==1.8.0
torchvision==0.9.0
numpy==1.21.2
scikit-learn==1.0.1
scipy==1.7.2
pandas==1.3.4
tqdm==4.63.0
xlwt==2.5
```
The environment with required packages can be created directly by running the following command:
```
conda env create -f environment.yml
```

## Datasets
For the **UCF-Crime** and **XD-Violence** datasets, we use off-the-shelf features extracted by [Wu et al](https://github.com/Roc-Ng). For the **ShanghaiTech** dataset, we used this [repo](https://github.com/v-iashin/video_features) to extract I3D features (highly recommended:+1:).
| Dataset     | Origin Video   | I3D Features  |
| -------- | -------- | -------- |
| &nbsp;&nbsp;UCF-Crime | &nbsp;&nbsp;[homepage](https://www.crcv.ucf.edu/projects/real-world/) | [download link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc) |
| &nbsp;XD-Violence | &nbsp;&nbsp;[homepage](https://roc-ng.github.io/XD-Violence/) | [download link](https://roc-ng.github.io/XD-Violence/) |
| ShanghaiTech | &nbsp;&nbsp;[homepage](https://svip-lab.github.io/dataset/campus_dataset.html) | [download link](https://drive.google.com/file/d/1kIv502RxQnMer-8HB7zrU_GU7CNPNNDv/view?usp=drive_link) |
| UBnormal | &nbsp;&nbsp;[homepage](https://github.com/lilygeorgescu/ubnormal#download) | [download link]([https://drive.google.com/file/d/1KbfdyasribAMbbKoBU1iywAhtoAt9QI0/view]) |

## Acknowledgement
Our codebase mainly refers to [STG-NF](https://github.com/orhir/STG-NF) and [PEL4VAD](https://github.com/yujiangpu20/PEL4VAD). We greatly appreciate their excellent contribution with nicely organized code!
