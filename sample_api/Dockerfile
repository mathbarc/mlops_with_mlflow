FROM python:3.11.13-bookworm as build


RUN pip install python-dotenv numpy torch opencv-python Flask mlflow[extras]==2.19.0 flask-cors torchvision
RUN apt-get update -y; apt-get install libopencv-dev -y

FROM python:3.11.13-bookworm as runner

COPY --from=build /usr/ /usr/
WORKDIR /home/
COPY rice_classify_api.py app.py
COPY rice_classifier.py rice_classifier.py
ENTRYPOINT [ "python3", "app.py" ]
