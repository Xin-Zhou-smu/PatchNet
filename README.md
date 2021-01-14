


# PatchNet: Hierarchical Deep Learning-Based Stable Patch Identification for the Linux Kernel [[pdf](https://arxiv.org/pdf/1911.03576.pdf)]

This repository is the newest version of PatchNet code bases.

## Code Structure
 - deeplearning:     contains all code files that relate to the training, validation and testing of PatchNet.
 - preprocessing:   contains code files that relate to the preprocessing of patches. Also useful if you want to use PatchNet on your own data.

## Reproduce the Results

Please see the README in the deeplearning folder to see how to download the data (pretrained models) and run the tool.

## Utilize PatchNet on New Dataset

Please see the README in the preprocessing folder to follow the steps we preprocess patches.
Then move the built new dataset into deeplearning folder and follow the README in the deeplearning folder to change the dataset format, train and test on PatchNet.

## Contact

Questions and discussion are welcome: vdthoang.2016@smu.edu.sg Or xinzhou.2020@phdcs.smu.edu.sg
