python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 512 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base1/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 1 \
    --logfile $1 \
    --fp16

python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 10 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 512 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base2/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 1 \
    --logfile $1 \
    --fp16

python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 40 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 512 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base3/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 2 \
    --logfile $1 \
    --fp16 

python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 256 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base4/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 1 \
    --logfile $1 \
    --fp16

python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 256 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base5/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 1 \
    --logfile $1 \
    --fp16

python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 256 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base6/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 2 \
    --logfile $1 \
    --fp16

python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 64 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base7/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 1 \
    --logfile $1 \
    --fp16

python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 64 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base8/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 1 \
    --logfile $1 \
    --fp16

python -u run_babi.py \
    --bert_model bert-base-uncased \
    --do_train \
    --do_predict \
    --do_lower_case \
    --train_file babidata_real/qa3/train.json \
    --predict_file babidata_real/qa3/test.json \
    --train_batch_size 256 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0 \
    --max_seq_length 64 \
    --doc_stride 256 \
    --output_dir ./outputs_perf/babidata_real-base9/$(basename babidata_real/qa3)/ \
    --gradient_accumulation_steps 2 \
    --logfile $1 \
    --fp16