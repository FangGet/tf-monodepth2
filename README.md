# tf-monodepth2

This is tensorflow(unofficial) implementation for the method in

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)  
>
> [arXiv 2018](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="assets/tf-monodepth2.gif" alt="example input output gif" width="600" />
</p>

Code mainly based on [SFMLearner](https://github.com/tinghuiz/SfMLearner) and [SuperPoint](https://github.com/rpautrat/SuperPoint)

If you find this work useful in your research please consider citing author's paper:

```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  journal = {arXiv:1806.01260},
  year = {2018}
}
```
## Prerequisites
This codebase was developed and tested with Tensorflow 1.6.0, CUDA 8.0 and Ubuntu 16.04.

## Setup
Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda env create -f environment.yml
conda activate tf-monodepth2
```

## Preparing training data
In order to train the model using the provided code, the data needs to be formatted in a certain manner. 

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command
```bash
python data/prepare_train_data.py --dataset_dir=/path/to/raw/kitti/dataset/ --dataset_name='kitti_raw_eigen' --dump_root=/path/to/resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
```

## Training(Only Monocular)

First of all, set dataset/saved_log path at monodepth2_kitti.yml

```shell
python train_monodepth2.py train config/monodepth2_kitti.yml your_saved_model_name
```

<p align="center">
  <img src="assets/tf-monodepth2.png" alt="example training log image" width="600" />
</p>

## Testing
```
python train_monodepth2.py test config/monodepth2_kitti.yml your_pretrained_model_name
```

pretrained model download link: [monodepth2_416*128_mono](https://drive.google.com/file/d/1oALNcevZSEvDHkjF1NX1Jf7JExWW52k-/view)

## Reference Codes
- Monodepth2
  - https://github.com/nianticlabs/monodepth2

- SfmLearner
  - https://github.com/tinghuiz/SfMLearner

- SuperPoint
  - https://github.com/rpautrat/SuperPoint
  
- resnet-18-tensorflow
  - https://github.com/dalgu90/resnet-18-tensorflow

## TODO
- [x] Auto-Mask loss described in paper
- [x] ResNet-18 Pretrained Model code
- [x] Testing part
- [ ] stereo and mono+stereo training



