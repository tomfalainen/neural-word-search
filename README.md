# neural-word-search

This repository contains the code necessary to reproduce the experiments from the paper 

**[Neural Word Search in Historical Manuscript Collections](https://arxiv.org/abs/1812.02771)**,

[Tomas Wilkinson](http://user.it.uu.se/~tomwi522/),
Jonas Lindström\,
Anders Brun

The paper addresses the problem of **Segmentation-free Word Spotting in Historical Manuscripts**, where a computer detects words on a collection of manuscript pages and allows a user to search within them. 

We provide:

- [Trained models](#trained-models) 
- Instructions for [training a model](#training-a-model) and evaluating on the IAM and Washington datasets

If you find this code useful in your research, please cite:

```
@article{wilkinson2018neural,
  title={Neural Word Search in Historical Manuscript Collections},
  author={Wilkinson, Tomas and Lindstr{\"o}m, Jonas and Brun, Anders},
  journal={arXiv preprint arXiv:1812.02771},
  year={2018}
}
```

## Installation
The models are implemented in [PyTorch](https://pytorch.org/) using Python 2.7. To install Pytorch, it's easiest to follow the instructions on their website. To use this repository with the least amount of work, it's recommended to use the [Anaconda distribution](https://www.anaconda.com/download/).

Once pytorch is installed you will require a few additional dependecies that are installed with the commands 

```
pip install easydict 
conda install opencv
```

## Trained models
You can download models trained on each dataset by following [this](https://uppsala.box.com/s/jkdy015j18ke41ed9501kgr2j4zfc7t1) link and downloading the zip file corresponding to a particular dataset. 

## Evaluating a model

To evaluate a model you can run
```
python test.py -weights models/model_name 
```

Or for a Ctrl-F-Mini model, run
```
python test_dtp.py -weights models/model_name 
```

There are a few flags relevant to testing in train_opts.py, such as evaluating with 4 folds for the washington benchmarks.

## Training a model

To download the IAM datasets, go to [this website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) and register. Download and unpack the data by running these commands, but with your own username and passwords

```
mkdir -p data/iam
cd data/iam
wget --user user --password pass http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt
wget --user user --password pass http://www.fki.inf.unibe.ch/DBs/iamDB/data/forms/formsA-D.tgz
wget --user user --password pass http://www.fki.inf.unibe.ch/DBs/iamDB/data/forms/formsE-H.tgz
wget --user user --password pass http://www.fki.inf.unibe.ch/DBs/iamDB/data/forms/formsI-Z.tgz
wget http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
mkdir forms
tar -C forms -xzf formsA-D.tgz
tar -C forms -xzf formsE-H.tgz
tar -C forms -xzf formsI-Z.tgz
unzip largeWriterIndependentTextLineRecognitionTask.zip
cd ../../
```

Similarly, for the [Washington dataset](http://ciir.cs.umass.edu/downloads/gw/gw_20p_wannot.tgz) run

```
mkdir -p data/washington/
cd data/washington
wget http://ciir.cs.umass.edu/downloads/gw/gw_20p_wannot.tgz
tar -xzf gw_20p_wannot.tgz
cd ../../
```

Next, you can either pre-augment the datasets and save them to H5 data files, which is quicker if training multiple times on a dataset, or you can skip this step and go directly to training while doing augmentation on the fly, which will be slower if training multiple models (unless you have lots of cpu cores). As of now, training ctrlfnet-mini using on the fly augmentation is not recommended, as it's quite slow to extract dtp proposals every training iteration.

Run 

```
python preprocess_h5.py -dataset washington -augment 1
python preprocess_h5.py -dataset iam -augment 1
```

You are now ready to train a model from scratch. However, we pre-trained models on the IIIT-HWS 10K dataset that we used for initialization. You can download models pretrained on the IIIT-HWS dataset [here](https://uppsala.box.com/s/rx5fm0s7m5q5lpk9wgkvirjc17xkjhgs) and unzip them in the directory models, but you may also train a model yourself by running 

```
mkdir -p data/iiit_hws
cd data/iiit_hws
wget http://ocr.iiit.ac.in/data/dataset/iiit-hws/iiit-hws.tar.gz
wget http://ocr.iiit.ac.in/data/dataset/iiit-hws/groundtruth.tar.gz
tar -xzf groundtruth.tar.gz
tar -xzf iiit_hws.tar.gz
cd ../../
python train.py -embedding dct -dataset iiit_hws 
```

Since this dataset only consists of segmented word images, we can only do full-page augmentation with it. As such we right now only support on the fly augmentation.

To train a model run 
```
mkdir -p checkpoints/ctrlfnet
mkdir -p checkpoints/ctrlfnet_mini
python train.py -dataset iam -save_id test -weights models/ctrlfnet_dct_iiit_hws.pt
```

To train a model run with a preprocessed h5 dataset add the h5 flag 
```
python train.py -dataset iam -save_id test -h5 1
python train.py -dataset washington -save_id test -h5 1
```

To train a Ctrl-F-Mini model make sure to use the train_dtp.py  file, for example:
```
python train_dtp.py -dataset washington -save_id test -h5 1 -weights models/ctrlfnet_mini_dct_iiit_hws.pt
```
