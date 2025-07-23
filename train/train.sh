#!/bin/bash
kaggle datasets download -d muratkokludataset/rice-image-dataset --unzip
python3 train_rice_classifier.py ./Rice_Image_Dataset/
