FROM ubuntu:18.04

RUN apt update
RUN apt install -y python3
RUN apt install -y python3-pip

# temporary until reqs calm down
RUN pip3 install tensorflow pillow matplotlib

ADD . /opt/learn

WORKDIR /opt/learn

RUN pip3 install -r requirements.txt