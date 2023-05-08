import os
from torch.utils.data import DataLoader
import torchvision

from cvat_sdk import make_client
from cvat_sdk.pytorch import ProjectVisionDataset, ExtractSingleLabelIndex

import train_rice_classifier
import rice_classifier



# log into the CVAT server
with make_client(host=os.environ["CVAT_ENDPOINT"], credentials=(os.environ["CVAT_USER"], os.environ["CVAT_PASS"])) as cvat_client:
    # get the dataset comprising all tasks for the Validation subset of project 12345
    dataset_train = ProjectVisionDataset(cvat_client, project_id=1,
        include_subsets=['Train'],
        # use transforms that fit our neural network
        transform=torchvision.transforms.ToTensor(),
        target_transform=ExtractSingleLabelIndex())
    
    dataset_test = ProjectVisionDataset(cvat_client, project_id=1,
        include_subsets=['Test'],
        # use transforms that fit our neural network
        transform=torchvision.transforms.ToTensor(),
        target_transform=ExtractSingleLabelIndex())

    # print the number of images in the dataset (in other words, the number of frames
    # in the included tasks)
    print(len(dataset_train))
    print(len(dataset_test))
    

    # params = {

    #     "lr": 0.0001,
    #     "momentum": 0.8,
    #     "batch_size": 12,
    #     "criterion": "cross_entropy",
    #     "optmizer": "sgd",
    #     "model": "rice_classifier_v1",
    #     "epochs": 10
    # }

    params = {

        "lr": 0.001,
        "momentum": 0.8,
        "batch_size": 64,
        "criterion": "cross_entropy",
        "optmizer": "sgd",
        "model": "rice_classifier_v1",
        "epochs": 20
    }
    

    labels = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

    trainLoader = DataLoader(dataset_train, params["batch_size"], True)
    testLoader = DataLoader(dataset_test, params["batch_size"], True)

    model = rice_classifier.RiceClassifierV1(labels)
    train_rice_classifier.train(trainLoader, testLoader, model, params, labels)


    

    ...