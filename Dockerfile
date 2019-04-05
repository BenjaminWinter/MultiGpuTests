FROM anibali/pytorch:cuda-10.0

ADD ./tests/ /tests

RUN sudo apt-get update && sudo apt-get install -y build-essential

RUN pip install pytorch-pretrained-BERT joblib tqdm

WORKDIR /tests

CMD ["bash", "./gputest.sh"]
