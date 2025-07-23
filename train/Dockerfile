FROM python:3.11.13-bookworm

RUN pip install python-dotenv numpy torch torchvision mlflow[extras]==2.19.0 kaggle opencv-python-headless
RUN apt update -y; apt install libopencv-dev -y

WORKDIR /home

COPY rice_classifier.py rice_classifier.py
COPY dataset_loader.py dataset_loader.py
COPY train_rice_classifier.py train_rice_classifier.py
COPY train.sh train.sh

CMD [ "bash", "train.sh" ]
