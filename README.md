
### **Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

[Hashmat Shadab Malik](https://scholar.google.com/citations?user=2Ft7r4AAAAAJ&hl=en), 
[Shahina Kunhimon](https://github.com/ShahinaKK),
[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Salman Khan](https://salman-h-khan.github.io),
and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

#

> **Abstract:** *Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch  under the condition of no labels and few data samples. Our training approach is based on min-max objective which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to our adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner.  We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection and video segmentation.* 

![main figure](images/Algo.png)

The Algorithm describes the training mechanism for training surrogate autoencoders. The equations in the above algorithm are mentioned in the paper.

## Contents
1) [Contributions](#Contributions) 
2) [Installation](#Installation)
3) [Dataset-Preparation](#Dataset-Preparation)
4) [Training](#Training)
5) [Attack](#Attack)
6) [Pretrained-Models](#Pretrained-Models)
7) [Results](#Results)
## Contributions
1. We propose self-supervised Adversarial Pixel Restoration to find highly transferable patterns by learning over flatter loss surfaces. Our training approach allows launching cross-domain attacks without access to large-scale labeled data or pretrained models.
2. Our proposed adversarial attack is self-supervised in nature and  independent of any task-specific objective. Therefore our approach can transfer perturbations to a variety of tasks as we demonstrate for classification, object detection, and segmentation.

<hr>
<hr>

## Installation
<sup>([top](#contents))</sup>
1. Create conda environment
```shell
conda create -n apr
```
2. Install PyTorch and torchvision
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
3. Install other dependencies
```shell
pip install -r requirements.txt
```

<hr />
<hr>

## Dataset-Preparation
<sup>([top](#contents))</sup>
**In-Domain Setting:** 5000 images are selected from ImageNet-Val (10 each from the first 500 classes).
Each surrogate model is trained only on few data samples e.g., 20 samples(default). Download the [ImageNet-Val](http://image-net.org/) classification dataset and structure the data as follows:
```
└───data
    ├── selected_data.csv
    └── ILSVRC2012_img_val
        ├── n01440764
        ├── n01443537
        └── ...
    
```
The `selected_data.csv` is used by the `our_dataset.py` to load the selected 5000 images from the dataset.
<hr>

**Cross-Domain Setting:** A single surrogate model is trained on large unannotated datasets. We use the following datasets for 
training:
  * [Paintings](https://www.kaggle.com/c/painter-by-numbers)
  * [Comics](https://www.kaggle.com/cenkbircanoglu/comic-books-classification)
  * [CoCo-2017(41k)](https://cocodataset.org/#download)

Directory structure should look like this:
 ```
    |paintings
        |images
                img1
                img2
                ...
```
<hr />
<hr>

## Training
**In-Domain Setting:** Each surrogate model is trained only on a few data samples
(20 by default). The model is trained by incorporating adversarial pixel transformation
based on rotation or jigsaw in an unsupervised setting. Supervised prototypical training mentioned in
this [paper]() is also trained in an adversarial fashion.

For training surrogate models with transformation:
1. _Rotation_
```shell
python train_id.py --mode rotate --n_imgs 20 --adv_train True --fgsm_step 2 \
--n_iters 2000 --save_dir ./trained_models
```
2. _Jigsaw_
```shell
python train_id.py --mode jigsaw --n_imgs 20 --adv_train True --fgsm_step 2 \
--n_iters 5000 --save_dir ./trained_models
```
3. _Prototypical_
```shell
python train_id.py --mode prototypical --n_imgs 20 --adv_train True --fgsm_step 2 \
--n_iters 15000 --save_dir ./trained_models
```
With 20 images used for training each surrogate model, overall 250 models would 
be trained for the selected 5000 ImageNet-Val images. The models would be saved
in like:
 ```
    |trained_models
        |models
                rotate_0.pth
                rotate_1.pth
                ...
```
<hr>

**Cross-Domain Setting:** A single surrogate model is trained adversarially on a large unannotated data
in an unsupervised setting by using rotation or jigsaw as pixel transfromations. 

For training the single surrogate model with transfromation:
1. _Rotation_
```shell
python train_cd.py  --mode rotate --adv_train True --fgsm_step 2 \
--end_epoch 50 --data_dir paintings/ --save_dir ./single_trained_models
```
2. _Jigsaw_
```shell
python train_cd.py  --mode jigsaw --adv_train True --fgsm_step 2 \
--end_epoch 50 --data_dir paintings/ --save_dir ./single_trained_models
```
change the `--data_dir` accordingly to train on comics, coco and any other dataset.
Setting `--adv_train` flag to False would result in the surrogate models trained 
by the baseline method mentioned in this [paper]().
<hr />
<hr>

## Attack
**In-Domain Setting:** For crafting adversarial examples on the selected 5000 
ImageNet-Val images, each trained surrogate model is used to mount an attack on the same 
set of images(default 20) on which it was trained. An L_inf based attack is run using:
```shell
python attack.py --epsilon 0.1 --ila_niters 100 --ce_niters 200 \
--ce_epsilon 0.1 --ce_alpha 1.0 --n_imgs 20 --ae_dir ./trained_models \
--mode rotate  --save_dir /path/to/save/adv_images
```
mode can be set as `rotate/jigsaw/prototypical` based on how the surrogate models
were trained. For `rotation/jigsaw` we can use a fully-unsupervised attack by 
passing `--loss unsup` as argument to the `attack.py` file.
<hr>

**Cross-Domain Setting:** A single surrogate model trained on a cross-domain dataset as
mentioned in the Training section is used to craft adversarial examples on 
the selected 5000 ImageNet-Val images. An L_inf based unsupervised attack is run using:
```shell
python attack.py --epsilon 0.1 --ila_niters 100 --ce_niters 200 \
--ce_epsilon 0.1 --ce_alpha 1.0 --n_imgs 20  --single_model True \
--chk_pth path/to/trained/model/weights.pth --save_dir /path/to/save/adv_images
```
<hr>
<hr>

### Pretrained-Models
**In-Domain Setting:** Pretrained weights for surrogate models trained 
with rotation/jigsaw/prototypical modes can be found [here]().

**Cross-Domain Setting:**
1. Models trained with rotation mode.

| Dataset   |                                               Baseline                                               |                                                                                             Ours | 
|:----------|:----------------------------------------------------------------------------------------------------:|-------------------------------------------------------------------------------------------------:|
| CoCo      |   [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/rotate_50_baseline_coco.pth)    |      [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/rotate_50_ours_coco.pth) | 
| Paintings | [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/rotate_50_baseline_paintings.pth) | [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/rotate_50_ours_paintings.pth) | 
| Comics    |  [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/rotate_50_baseline_comics.pth)   |    [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/rotate_50_ours_comics.pth) | 


2. Models trained with jigsaw mode.

| Dataset   |                                               Baseline                                               |                                                                                             Ours | 
|:----------|:----------------------------------------------------------------------------------------------------:|-------------------------------------------------------------------------------------------------:|
| CoCo      |   [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/jigsaw_50_baseline_coco.pth)    |      [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/jigsaw_50_ours_coco.pth) | 
| Paintings | [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/jigsaw_50_baseline_paintings.pth) | [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/jigsaw_50_ours_paintings.pth) | 
| Comics    |  [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/jigsaw_50_baseline_comics.pth)   |    [Link](https://github.com/HashmatShadab/APR/releases/download/v1.0/jigsaw_50_ours_comics.pth) | 


<hr />
<hr />

## Results
We compare transferability of surrogate models trained by our approach with the 
approach followed by the baseline -> [Practical No-box Adversarial Attacks (NeurIPS-2021)](https://arxiv.org/abs/2012.02525).
After generating adversarial examples on the selected 5000 ImageNet-Val images,
we report the top-1 accuracy on several classification 
based models _(lower is better)_.


**In-Domain Setting:**

1. Accuracy on Convolutional Networks.
![results](images/Table1.png)

2. Accuracy on Vision Transformers.
![results](images/Table2.png)

**Cross-Domain Setting:**

<hr />
<hr />

## Citation
If you use our work, please consider citing:
```bibtex
   
```

<hr />

## Contact
Should you have any question, please create an issue on this repository or contact at hashmat.malik@mbzuai.ac.ae

<hr />

## References
Our code is based on [ Practical No-box Adversarial Attacks against DNNs](https://github.com/qizhangli/nobox-attacks) repository. 
We thank them for releasing their code.
