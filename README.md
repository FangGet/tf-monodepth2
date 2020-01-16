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



<b>Update: fix testing bug for input data normalization, thanks @JiatianWu </b>



<b>Current pretrained model result is slightly low than paper.</b>



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

## Eval Result (with pretrained model link)
|  model_name  | abs_rel | sq_rel | rms | log_rms | δ<1.25 | δ<1.25^2 | δ<1.25^3 |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
| mono_640x192_nopt(paper) | 0.132 | 1.044 | 5.142 | 0.210 | 0.845 | 0.948 | 0.977 |
| [mono_640x192_nopt(ours)](https://drive.google.com/file/d/13jYuDrHiK9uoRmu1rXUxSBv-yEx6tzWJ/view?usp=sharing) | 0.139 | 1.1293 | 5.4190 | 0.2200 | 0.8299 | 0.9419 | 0.9744 |
| `mono_640x192_pt(paper)` | 0.115 | 0.903 | 4.863 | 0.193 | 0.877 | 0.959 | 0.982 |
| [mono_640x192_pt(ours)](https://drive.google.com/file/d/1Bk9gMrzuF_QrDRv11ILrqv3xHvqxZR2a/view?usp=sharing) | 0.120 | 0.8702 | 4.888 | 0.194 | 0.861 | 0.957 | 0.982 |


## Demonstration (Click Image For Youtube Video)
<p align="center">
<a href="https://www.youtube.com/watch?v=TUgaPZgdEys
" target="_blank"><img src="assets/depth_start.png"
alt="demo for tf-monodepth2" width="720" height="540" /></a>
</p>





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
python monodepth2.py train config/monodepth2_kitti.yml your_saved_model_name
```

<p align="center">
  <img src="assets/tf-monodepth2.png" alt="example training log image" width="600" />
</p>

## Testing
```
python monodepth2.py test config/monodepth2_kitti.yml your_pretrained_model_name
```

<font color="red">Get pretrained model download link from Eval Result chart</font>

## Evaluation
* First we need to save predicted depth image into npy file
```
python monodepth2.py eval config/monodepth2_kitti_eval.yml your_pretrained_model_name depth
```
Save destination should be setted in monodepth2_kitti_eval.yml.
* Then we use evaluation code to compute error result:
```
cd kitti_eval
python2 eval_depth.py --kitti_dir=/your/kitti_data/path/ --pred_file=/your/save/depth/npy/path/ --test_file_list=../data/kitti/test_files_eigen.txt
```
Note: please use python2 to execute this bash.

<font color="red">kitti_eval code from Zhou's SFMLearner</font>

Pose evaluation code to be completed.

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
- [ ] Evaluation for pose estimation
- [ ] stereo and mono+stereo training



