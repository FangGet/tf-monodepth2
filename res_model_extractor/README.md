# ResNet-18 Model Extractor (From Torch)

This pretrained model is extracted from pytorch, please refer to:

```
https://github.com/dalgu90/resnet-18-tensorflow
```

## Prerequisite

* numpy
* pickle
* torchfile



## How To Run

* convert torch .t7 into numpy dictionary file:

```
# Download the ResNet-18 torch checkpoint
wget https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
# Convert into tensorflow checkpoint
python extract_torch_t7.py
```



