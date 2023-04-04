# Intel's JPQD MLPerf Bert Submission

This folder provides steps to obtain JPQD set of models. For description of JPQD, please refer [open/Intel/code/bert-99.9_jpqd-large/openvino-cpu](/open/Intel/code/bert-99.9_jpqd-large/openvino-cpu)

## Setup
```bash
# Create conda environment
ENVNAME=optimum-intel-openvino
conda create -n $ENVNAME python=3.8
conda activate $ENVNAME

# Install Optimum-Intel/OpenVINO
git clone https://github.com/huggingface/optimum-intel
cd optimum-intel
git checkout 791e8c66 -b mlperfv3-jpqd
pip install -e .[openvino,nncf]
pip install tensorboard

# Install Question Answering Dependencies
cd examples/openvino/question-answering
pip install -r requirements.txt
```
## JPQD Training
**IMPORTANT:** The following commands rely on the content of this folder, please ensure this repository is cloned. When running commands following snippets below, please update the variables accordingly.
### 1. JPQD-BERT-large-99.9
```bash
RUNID=jpqd-bert-large-99.9
NNCFCFG=/path/to/configs/jpqd-bert-large-99.9-r0.120.json
OUTDIR=/path/to/save/checkpoint
MASTER_PORT=13500

cd optimum-intel/examples/openvino/question-answering

# use 4 GPUs to train
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port $MASTER_PORT \
    run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --dataset_name squad \
    --teacher_model_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --distillation_weight 0.9 \
    --do_eval \
    --fp16 \
    --do_train \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 500 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR \
    --nncf_compression_config $NNCFCFG \
```
```bash
# Find best checkpoint in $OUTDIR and reshape the OV model for 384 in sequence length
CKPT_XML=/path/to/outdir/checkpoint/openvino_model.xml
python3 reshape_ir_seqlen_1x384.py $CKPT_XML bert-99.9_jpqd-large/openvino_model.xml
```

### 2. JPQD-BERT-large-99
```bash
# Run command as above with the following configurations
RUNID=jpqd-bert-large-99.9
NNCFCFG=/path/to/configs/jpqd-bert-large-99-r0.160.json
OUTDIR=/path/to/save/checkpoint
MASTER_PORT=13600
```
```bash
# Find best checkpoint in $OUTDIR and reshape the OV model for 384 in sequence length
CKPT_XML=/path/to/outdir/checkpoint/openvino_model.xml
python3 reshape_ir_seqlen_1x384.py $CKPT_XML bert-99_jpqd-large/openvino_model.xml
```
### 3. JPQD-BERT-base-99
```bash
RUNID=jpqd-bert-base-99
NNCFCFG=/path/to/configs/jpqd-bert-base-99-r0.015.json
OUTDIR=/path/to/save/checkpoint

cd optimum-intel/examples/openvino/question-answering

# Use a single GPU to train
python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --teacher_model_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --distillation_weight 0.9 \
    --do_eval \
    --fp16 \
    --do_train \
    --learning_rate 3e-5 \
    --num_train_epochs 15 \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 500 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR \
    --nncf_compression_config $NNCFCFG \
```
```bash
# Find best checkpoint in $OUTDIR and reshape the OV model for 384 in sequence length
CKPT_XML=/path/to/outdir/checkpoint/openvino_model.xml
python3 reshape_ir_seqlen_1x384.py $CKPT_XML bert-99_jpqd-base/openvino_model.xml
```
### 4. JPQD-MobileBERT-99

```bash
# Apply patch to optimum-intel
cd optimum-intel/examples/openvino/question-answering
git apply /path/to/optimum-intel-changes-to-enable-jpqd-on-mobilebert.patch
```
```bash
RUNID=jpqd-mobilebert-99
NNCFCFG=configs/jpqd-mobilebert-99-r0.020.json
OUTDIR=/path/to/save/checkpoint

cd optimum-intel/examples/openvino/question-answering

# Use a single GPU to train
python run_qa.py \
    --dataset_name squad \
    --model_name_or_path google/mobilebert-uncased \
    --num_tx_block 15 \
    --teacher_model_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --distillation_weight 0.9 \
    --distillation_temperature 2 \
    --do_eval \
    --do_train \
    --fp16 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --optim adamw_torch \
    --num_train_epochs 16 \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 32 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_steps 500 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --nncf_compression_config $NNCFCFG \
    --run_name $RUNID \
    --output_dir $OUTDIR \
```
```bash
# Find best checkpoint in $OUTDIR and reshape the OV model for 384 in sequence length
CKPT_XML=/path/to/outdir/checkpoint/openvino_model.xml
python3 reshape_ir_seqlen_1x384.py $CKPT_XML bert-99_jpqd-mobilebert/openvino_model.xml
```