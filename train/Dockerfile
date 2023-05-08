FROM python:3.8.2-slim

RUN pip install python-dotenv
RUN pip install numpy
RUN pip install torch
RUN pip install torchvision
RUN pip install mlflow[extras]
RUN pip install cvat-sdk


WORKDIR /home

COPY rice_classifier.py rice_classifier.py
COPY train_rice_classifier.py train_rice_classifier.py
COPY train_with_cvat_dataset.py train_with_cvat_dataset.py

ENTRYPOINT [ "python", "train_with_cvat_dataset.py" ]