import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import numpy
import tqdm

import mlflow
import os
import dotenv

dotenv.load_dotenv(".env")


def train(
    train_loader,
    test_loader,
    net,
    params,
    labels_str,
    use_cuda=False,
):
    mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
    experiment = mlflow.get_experiment_by_name("Rice Classifier")
    if experiment is None:
        experiment_id = mlflow.create_experiment("Rice Classifier")
    else:
        experiment_id = experiment.experiment_id
    mlflow.start_run(experiment_id=experiment_id)

    params["model_id"] = net.id

    mlflow.log_params(params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=params["lr"], momentum=params["momentum"]
    )

    device_str = "cpu"
    if torch.cuda.is_available() and use_cuda:
        device_str = "cuda"

    device = torch.device(device_str)
    net = net.to(device)

    for epoch in tqdm.tqdm(
        range(params["epochs"])
    ):  # loop over the dataset multiple times
        measures = {}

        net.train(True)
        count = 0
        train_loss = 0
        for data in train_loader:
            # get the input images and their corresponding labels
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # forward pass to get outputs
            outputs = net(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # backward pass to calculate the parameter gradients
            loss.backward()

            train_loss += loss.item()

            # update the parameters
            optimizer.step()
            count += 1
            if count == 30:
                train_loss = train_loss / 30
                break

        measures["train_loss"] = train_loss
        net.eval()

        class_correct = numpy.zeros(len(labels_str))
        class_total = numpy.zeros(len(labels_str))
        test_loss = 0

        count = 0
        for data in test_loader:
            # get the input images and their corresponding labels
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass to get outputs
            outputs = net(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # update average test loss
            test_loss = test_loss + loss.item()

            # get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs.data, 1)

            # compare predictions to true label
            correct = numpy.squeeze(predicted.eq(labels.data.view_as(predicted)))

            # calculate test accuracy for *each* object class
            for i in range(len(labels)):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

            count += 1
            if count == 30:
                test_loss = test_loss / 30
                break

        measures["test_loss"] = test_loss

        for i in range(len(labels_str)):
            if class_total[i] > 0:
                measures[f"test_accuracy_{labels_str[i]}"] = (
                    class_correct[i] / class_total[i]
                )

        measures["test_accuracy_overall"] = numpy.sum(class_correct) / numpy.sum(
            class_total
        )

        mlflow.log_metrics(measures, epoch)

        if epoch % 10 == 0:
            mlflow.pytorch.log_model(
                net,
                f"rice_classifier_{epoch}",
                signature=net.signature(),
                code_paths=["train/rice_classifier.py"],
            )

    mlflow.pytorch.log_model(
        net,
        "rice_classifier_final",
        signature=net.signature(),
        code_paths=["train/rice_classifier.py"],
    )


if __name__ == "__main__":
    import sys
    import dataset_loader
    import rice_classifier

    from torch.utils.data import DataLoader

    import os

    params = {
        "lr": 0.0001,
        "momentum": 0.8,
        "batch_size": 12,
        "criterion": "cross_entropy",
        "optmizer": "sgd",
        "model": "rice_classifier_v1",
        "epochs": 20,
    }

    # params = {

    #     "lr": 0.001,
    #     "momentum": 0.8,
    #     "batch_size": 20,
    #     "criterion": "cross_entropy",
    #     "optmizer": "sgd",
    #     "model": "rice_classifier_v1",
    #     "epochs": 30
    # }

    labels = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

    trainData, testData = dataset_loader.ImageClassificationDataset(
        sys.argv[1], torchvision.transforms.ToTensor()
    ).split()
    trainLoader = DataLoader(trainData, params["batch_size"], True)
    testLoader = DataLoader(testData, params["batch_size"], True)

    model = rice_classifier.RiceClassifierV1(labels)

    train(trainLoader, testLoader, model, params, labels)
