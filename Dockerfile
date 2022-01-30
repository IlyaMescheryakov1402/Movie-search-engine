FROM ubuntu:latest

ENV TZ=Europe/Moscow

EXPOSE 5000

COPY . /docker
WORKDIR /docker
RUN apt-get update
RUN apt-get install -y python3.8
RUN echo "python installed"
RUN apt-get install -y pip
RUN echo "pip installed"
RUN pip install -r requirements.txt 
RUN echo "requirements installed"
CMD ["python3.8", "server.py"]