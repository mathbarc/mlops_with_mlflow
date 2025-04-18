#!/bin/bash

kaggle datasets download -d muratkokludataset/rice-image-dataset --unzip
python3 split_dataset.py

# cd Rice_Image_Dataset/train
# zip -r ../../train.zip .
#
# cd ../test
# zip -r ../../test.zip .
