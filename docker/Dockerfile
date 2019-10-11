
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL maintainer="kye shimizu"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt update --fix-missing && \
    apt install -y --no-install-recommends wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion unzip libjpeg-dev libpng-dev build-essential vim nodejs && \
    apt -y autoremove && apt -y autoclean &&\
    #making python3 default
    rm -f /usr/bin/python && ln -s /usr/bin/python /usr/bin/python3


RUN rm -rf /var/cache/apt

CMD [ "/bin/bash" ]

ENV PATH /usr/local/cuda-10.0/bin${PATH:+:${PATH}}

#install stuff in requirements.txt
RUN git clone --recurse-submodules https://github.com/abraham-ai/eden.git /eden
WORKDIR /eden

RUN pip install -r requirements.txt &&\
   python setup_external_libs.py &&\
   jupyter labextension install @jupyter-widgets/jupyterlab-manager &&\