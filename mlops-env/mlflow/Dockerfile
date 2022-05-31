FROM python:3.8.2-slim

RUN pip install PyMySQL && \   
    pip install psycopg2-binary && \
    pip install mlflow[extras]

ENTRYPOINT ["mlflow"]