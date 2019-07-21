FROM python:2.7.15

MAINTAINER Gili Karni "gili@minerva.kgi.edu"

USER root

# Set the working directory to /home
WORKDIR /home

RUN apt-get -qq update

RUN apt-get install -y \

        locales \
        wget \
        python2.7 \
        tar \
        sudo \
        vim \
        git \
        python2.7-dev \
        gnome-devel \
        libcr-dev \
        mpich \
        mpich-doc \
        libncursesw5-dev \
        ncurses-base \
        ncurses-bin \
        ncurses-term



### install interviews ###
RUN wget https://neuron.yale.edu/ftp/neuron/versions/v7.6/iv-19.tar.gz
RUN tar zxf iv-19.tar.gz
RUN mv iv-19 iv

WORKDIR ./iv
RUN sh ./build.sh
RUN ./configure --prefix='/home/iv'
RUN make
RUN make install


### install neuron ###
RUN wget https://neuron.yale.edu/ftp/neuron/versions/v7.6/7.6.2/nrn-7.6.2.tar.gz
RUN tar zxf nrn-7.6.2.tar.gz
RUN mv nrn-7.6 nrn

WORKDIR ./nrn
RUN ./configure --prefix='/home/nrn/' --with-iv='/home/iv' --with-nrnpython='/usr/local/bin/python' --with-paranrn
RUN make
RUN make install


### add to path ###
ENV PYTHONPATH=/usr/local/bin/python
ENV PATH=/home/iv/x86_64/bin:/home/nrn/x86_64/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/home/nrn/x86_64/lib:/home/iv/x86_64/lib

RUN pip install scipy numpy matplotlib cython mpi4py neuronpy netpyne

ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python2.7/dist-packages



### utils ###
# modify plotting back-end
ENV MPLBACKEND="agg"

WORKDIR /home
RUN adduser --disabled-password --gecos '' notroot
