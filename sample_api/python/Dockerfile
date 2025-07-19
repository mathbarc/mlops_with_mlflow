FROM python:3.8.2-slim

RUN pip install python-dotenv numpy torch opencv-python Flask mlflow[extras] flask-cors
RUN apt-get update
RUN apt-get install libopencv-dev -y

WORKDIR /home/
COPY rice_classify_api.py app.py
COPY .env .env
ENTRYPOINT [ "python3", "app.py" ]
