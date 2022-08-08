FROM python:3.8.2-slim

RUN pip install python-dotenv
RUN pip install numpy
RUN pip install torch
RUN pip install opencv-python
RUN pip install Flask
RUN pip install mlflow[extras]

RUN apt update
RUN apt install libopencv-dev -y

COPY rice_classify_api.py app.py

ENTRYPOINT [ "flask", "run" ]