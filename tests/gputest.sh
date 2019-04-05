rm -r /pvc/results/gpu_perftests/
mkdir -p /pvc/results/gpu_perftests/

nvidia-smi > /pvc/results/gpu_perftests/smi.log
nvidia-smi topo -m > /pvc/results/gpu_perftests/topo.log
nvidia-smi -a > /pvc/results/gpu_perftests/nvidia_full.log

bash perftest_linear.sh /pvc/results/perftests/linear.log
bash perftest_lstm.sh /pvc/results/perftests/lstm.log
bash perftest_bert_2gpu.sh /pvc/results/perftests/bert_fp32.log
bash perftest_bert_2gpu_fp16.sh /pvc/results/perftests/bert_fp16.log


