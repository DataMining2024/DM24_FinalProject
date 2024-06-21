FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV TZ Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install \
    python3 python3-pip python3-dev \
    libgl1 tzdata git wget ssh vim

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN echo "root:password" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

WORKDIR /app

COPY . /app/

RUN apt-get update && apt-get install -y \
    python3-pip \
    && apt-get clean

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py", "--mode", "train", "--config", "configs/config.json"]