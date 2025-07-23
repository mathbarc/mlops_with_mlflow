FROM nginx:alpine

ARG API_ENDPOINT
ENV API_ENDPOINT=${API_ENDPOINT}

COPY . /usr/share/nginx/html
RUN sed -i -e "s/http:\/\/localhost\:5000/${API_ENDPOINT:-http:\/\/localhost\:5000}/g" /usr/share/nginx/html/rice-mlops.js
