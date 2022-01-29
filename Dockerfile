FROM ubuntu:latest

ENV TZ=Europe/Moscow

COPY . /docker
WORKDIR /docker
# RUN ls
RUN apt-get update
RUN apt-get install -y python3.8
RUN echo "python installed"
RUN apt-get install -y pip
RUN echo "pip installed"
RUN pip install -r requirements.txt 
RUN echo "requirements installed"