#  Learning from Limited and Imperfect Data (L2ID): Classification Challenges


### Website
#### https://l2id.github.io/challenge_classification.html


### Paper

Please cite the following paper in use of this evaluation framework:

```
@inproceedings{guo2020broader,
  title={A broader study of cross-domain few-shot learning},
  author={Guo, Yunhui and Codella, Noel C and Karlinsky, Leonid and Codella, James V and Smith, John R and Saenko, Kate and Rosing, Tajana and Feris, Rogerio},
  year={2020},
  organization={ECCV}
}
```

## Introduction

We will have three tracks investigating *cross-domain*, *multi-source settings* as well as *discrimination across a larger number of classes*, bridging the gap between few-shot learning, domain adaptation, and semi-supervised learning. All tracks will include multiple sources and the use of unlabeled data to support semi-supervised algorithms. 


# <span style="color:red">Track 1: Cross-domain, *small* scale</span>

## Description
This setting is similar as the previous VL3 workshop challenge (https://www.learning-with-limited-labels.com/challenge), supporting teams that would like to continue development for cross-domain few-shot learning. However, we will have <b>multiple</b> sources rather than relying on ImageNet solely, with no explicit label overlap between sources and target. These additional sources remain consistent with prior literature, allowing results to be directly comparable to prior results (https://arxiv.org/abs/1912.07200)

## Datasets
The following datasets are used for evaluation in this challenge:

### Source domain: 

* miniImageNet 
* CUB (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CIFAR100
* Caltech256
* DTD (<https://www.robots.ox.ac.uk/~vgg/data/dtd/>)

### Target domains: 

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.isic-archive.com/data#2018

* **Plant Disease**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data

    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`
    

# <span style="color:red">Track 2: Cross-domain, *Large* scale</span>

## Description
In this track we add additional datasets to both source and target datasets for participants with sufficient compute resources. Importantly, in this task, in addition to the multiple sources, we provide <b>multiple tasks</b> from which to draw source data or models. 

## Datasets
The following datasets are used for evaluation in this challenge:

### Source domain: 

* miniImageNet 
* CUB (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CIFAR100
* Caltech256
* DTD (<https://www.robots.ox.ac.uk/~vgg/data/dtd/>)
* DomainNet (http://ai.bu.edu/M3SDA/#dataset)
* COCO (2017 Train Images) (https://cocodataset.org/#download)
* PASCAL (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data)
* KITTI (http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
* Cityscapes (*any version or task)  (https://www.cityscapes-dataset.com/dataset-overview/#class-definitions)


### Target domains: 

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.isic-archive.com/data#2018

* **Plant Disease**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data

    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`

* **PatchCamelyon**:

    Home: https://github.com/basveeling/pcam (level_2_split_train)


* **KenyanFood13**:

    Home: https://drive.google.com/file/d/1CHnTy4XqGowT2chCBBxZ8rdYxMC1bwTc/view (training image)


* **IP102**:

    Home: https://github.com/xpwu95/IP102 (training image)
    

* **Bark-101**:

    Home: http://eidolon.univ-lyon2.fr/~remi1/Bark-101/ (training image) 



## General information

* **No meta-learning in-domain**
* 5-way classification
* n-shot, for varying n per dataset
* 100 randomly selected few-shot 5-way trials up to 50-shot (scripts provided to generate the trials)
* Average accuracy across all trials reported for evaluation.

* **For generating the trials for evaluation, please refer to finetune.py and the examples below**

## Specific Tasks:

### Tasks for Track 1 and Track 2

**EuroSAT**

  • Shots: n = {5, 20, 50}

**ISIC2018**

  • Shots: n = {5, 20, 50}

**Plant Disease**

  • Shots: n = {5, 20, 50}

**ChestX-Ray8**

  • Shots: n = {5, 20, 50}

### Additional Tasks for Track 2

**PatchCamelyon**

  • Shots: n = {5, 20, 50}

**KenyanFood13**

  • Shots: n = {5, 20, 50}
 
**IP102**

  • Shots: n = {5, 20, 50}
  
**Bark-101**

  • Shots: n = {5} 
  
 
## Enviroment

Python 3.5.5

Pytorch 0.4.1

h5py 2.9.0

## Steps

1. Download the datasets for specific tracks for evaluation using the above links.

2. Download miniImageNet using <https://drive.google.com/file/d/1uxpnJ3Pmmwl-6779qiVJ5JpWwOGl48xt/view?usp=sharing>

3. Change configuration file `./configs.py` to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to.

4. Train base models on miniImageNet

    • *Standard supervised learning on miniImageNet*

    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug
    ```

    • *Train meta-learning method (protonet) on miniImageNet*

    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method protonet --n_shot 5 --train_aug
    ```

5. Save features for evaluation (optional, if there is no need to adapt the features during testing) 

    • *Save features for testing*

    ```bash
        python save_features.py --model ResNet10 --method baseline --dataset CropDisease --n_shot 5 --train_aug
    ```

6. Test with saved features (optional, if there is no need to adapt the features during testing) 

    ```bash
        python test_with_saved_features.py --model ResNet10 --method baseline --dataset CropDisease --n_shot 5 --train_aug
    ```

7. Test

    • *Finetune with frozen model backbone*: 
 
    ```bash
        python finetune.py --model ResNet10 --method baseline  --train_aug --n_shot 5 --freeze_backbone
    ```

    • *Finetune*

    ```bash
        python finetune.py --model ResNet10 --method baseline  --train_aug --n_shot 5 
    ```
    
    • *Example output:* 600 Test Acc = 49.91% +- 0.44%

8. Test with Multi-model selection (make sure you have trained models on all the source domains (miniImageNet, CUB, Caltech256, CIFAR100, DTD))

    • *Test Multi-model selection without fine-tuning*: 
   
    ```bash
       python model_selection.py --model ResNet10 --method baseline  --train_aug --n_shot 5 
    ```

    • *Test Multi-model selection without fine-tuning*: 
  
     ```bash
       python model_selection.py --model ResNet10 --method baseline  --train_aug --n_shot 5 --fine_tune_all_models
     ```

9. For testing your own methods, simply replace the function **finetune()** in `finetune.py` with your own method. Your method should at least have the following arguments,

    • *novel_loader: data loader for the corresponding dataset (EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8 for track 1)*

    • *n_query: number of query images per class*

    • *n_way: number of shots*

    • *n_support: number of support images per class*



## References

[1] Sharada P Mohanty, David P Hughes, and Marcel Salathe. Using deep learning for image
based plant disease detection. Frontiers in plant science, 7:1419, 2016

[2] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel
dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of
Selected Topics in Applied Earth Observations and Remote Sensing , 12(7):2217–2226, 2019.

[3] Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The ham10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin lesions.
Scientific data, 5:180161, 2018.

[4] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M Emre Celebi, Stephen
Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael
Marchetti, et al. Skin lesion analysis toward melanoma detection 2018: A challenge
hosted by the international skin imaging collaboration (isic). arXiv preprint.
arXiv:1902.03368, 2019

[5] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald
M Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly
supervised classification and localization of common thorax diseases. In Proceedings of the
IEEE conference on computer vision and pattern recognition, pages 2097–2106, 2017

