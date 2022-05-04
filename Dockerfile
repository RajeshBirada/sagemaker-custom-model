FROM ubuntu:latest

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
#RUN ln -s /usr/bin/pip3 /usr/bin/pip





#FROM ubuntu:latest

# first layers should be dependency install so changes in code won't cause the build to
# start from scratch.
COPY requirements.txt /opt/program/requirements.txt
#RUN useradd -ms /bin/bash admin


RUN pip3 install --no-cache-dir -r /opt/program/requirements.txt

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
#ENV MODEL_PATH="/opt/ml/model"

#EXPOSE 8080
# Set up the program in the image
COPY model /opt/program
WORKDIR /opt/program

