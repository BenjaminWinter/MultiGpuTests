# Tests for Multi Gpu Clusters
This repo contains the means to build a docker image that does some sanity checks using nvidia-smi
 and 4 real-world performance tests.

## Sanity Tests
create logs of :

    -   nvidia-smi

    -   nvidia-smi topo -m

    -   nvidia-smi -a

### Performance Tests
4 different settings each with 9 combinations of batch and input sizes

    -   linear: runs vectors through a small MLP

    -   lstm: runs sequences through a multi-layer BiLSTM

    -   bert: runs bAbI sequences through a pretrained BERT model with Classification head

    -   bert_fp16: same as above, uses fp16 mode