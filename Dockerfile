FROM python:3.7.2

MAINTAINER Gili Karni "gili@minerva.kgi.edu"

USER root

# Set the working directory to /home
WORKDIR /home

RUN apt-get -qq update

RUN apt-get install -y \
    locales \
    wget \
    python3 \
    tar \
    sudo \
    vim \
    git \
    python3-dev \
    gnome-devel \
    libcr-dev \
    mpich \
    mpich-doc \
    libncursesw5-dev \
    ncurses-base \
    ncurses-bin \
    ncurses-term

ENV PYTHON3PATH=/usr/local/bin/python3 #'which python3'


### install MPIch ###
RUN wget http://www.mpich.org/static/downloads/3.3.1/mpich-3.3.1.tar.gz
RUN tar zxf mpich-3.3.1.tar.gz
RUN mv mpich-3.3.1 mpi
RUN mkdir mpi_build


WORKDIR ./mpi
RUN ../mpi/configure --prefix='$HOME/mpi_build'
RUN make
RUN make install

ENV PATH=$HOME/mpi/bin:$PATH
# test via mpiexec -n 2 echo 'hi'

### install interviews ###
RUN wget https://neuron.yale.edu/ftp/neuron/versions/v7.6/iv-19.tar.gz
RUN tar zxf iv-19.tar.gz
RUN mv iv-19 iv

WORKDIR ./iv
RUN sh ./build.sh
RUN ./configure --prefix='$HOME/iv'
RUN make
RUN make install


### install neuron ###
RUN wget https://neuron.yale.edu/ftp/neuron/versions/v7.6/7.6.2/nrn-7.6.2.tar.gz
RUN tar zxf nrn-7.6.2.tar.gz
RUN mv nrn-7.6 nrn

WORKDIR ./nrn
RUN ./configure --prefix='$HOME/nrn/' --with-iv='$HOME/iv' --with-nrnpython='$PYTHON3PATH' --with-paranrn
RUN make
RUN make install


### add to path ###
ENV PYTHONPATH=$PYTHON3PATH
ENV PATH=$HOME/iv/x86_64/bin:$HOME/nrn/x86_64/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$HOME/nrn/x86_64/lib:$HOME/iv/x86_64/lib/

RUN pip install scipy numpy matplotlib cython mpi4py neuronpy netpyne

ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3/dist-packages



### utils ###
# modify plotting back-end
ENV MPLBACKEND="agg"

WORKDIR /home
RUN adduser --disabled-password --gecos '' notroot
