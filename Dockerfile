FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7

RUN pip install pytorch-pretrained-BERT==0.6.1 joblib tqdm

ADD ./tests/ /tests

WORKDIR /tests

RUN git clone https://github.com/NVIDIA/apex.git

WORKDIR apex

RUN python setup.py install --cuda_ext --cpp_ext

WORKDIR /tests

CMD ["bash", "./gputest.sh"]
