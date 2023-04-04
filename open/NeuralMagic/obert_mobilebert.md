# oBERT-MobileBERT: The Optimal BERT Surgeon applied to the MobileBERT model

[The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models](https://arxiv.org/abs/2203.07259) (oBERT) is an efficient and accurate weight pruning method based on approximate second-order information, which we showed to yield state-of-the-art results in both stages of language tasks: pre-training and fine-tuning. For the MLPerf inference submission, we apply it in the fine-tuning stage on the SQuADv1.1 task. More specifically we adopt the gradual downstream pruning setup presented in the paper and progressively prune and fine-tune the popular [mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) model.

High-performance inference usually benefits more from (semi) structured sparsity patterns than from the unstructured ones. Hence, we employ direct layer-dropping, generalized oBERT formulation introduced in the paper which prunes weights in the 4-block pattern, meaning that contiguous blocks of 4 weights are either set to zero or kept dense. Both pruning types, unstructured and 4-block, can be leveraged for computational speedups with the DeepSparse runtime, but 4-block pruning coupled with INT8 quantization can provide further performance gains. For quantization, we apply standard quantization-aware training QAT on top of the 4-block pruned models.

To ease reproducibility, we conduct our experiments with popular open-source libraries: [Transformers](https://github.com/huggingface/transformers) and [SparseML](https://github.com/neuralmagic/sparseml). As previously noted, our compression setup consists of two steps, pruning and quantization, and now we present in detail each one of them with the corresponding configuration files (which we also call *compression recipes*) and bash scripts to reproduce our results.

Our compression setup with the MobileBERT model takes up to 6GB of GPU memory in total, which makes it easily reproducible on all modern GPUs.

## Step 1: Layer-Dropping and Semi-Structured Gradual Pruning

The encoder part of the MobileBERT architecture has 24 identical layers, each composed of several attention, bottleneck and fully-connected modules. Following insights about layer-dropping from the paper, we have identified that not all 24 layers are needed to successfully recover dense model's accuracy on the SQuADv1 task which is why we apply direct layer-dropping by keeping the first 14 layers and dropping the rest.
Then, we follow our gradual pruning setup to progressively prune the remaining 14 layers of MobileBERT model over the span of 30 training epochs. More specifically, we make use of knowledge-distillation from the dense teacher, learning rate scheduler with rewinds and cubic sparsity scheduler with high initial pruning step. We prune the encoder part in the semi-structured 4-block pattern up to 50% sparsity.

Assuming that the SparseML library is installed, the bash script to reproduce our pruning setup is as follows:
```shell
CUDA_VISIBLE_DEVICES=0 python src/sparseml/transformers/question_answering.py \
    --distill_teacher zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base-none \
    --model_name_or_path zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base-none \
    --dataset_name squad \
    --do_train \
    --fp16 \
    --do_eval \
    --optim adamw_torch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --preprocessing_num_workers 8 \
    --seed 42 \
    --num_train_epochs 30 \
    --recipe obert_mobilebert_compression_recipe.yaml \
    --output_dir my_pruning_output_dir
```

And the *obert_mobilebert_compression_recipe.yaml* is as follows:
```yaml
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: 30.0

  - !TrainableParamsModifier
    params:
    - mobilebert.embeddings.word_embeddings.weight
    - mobilebert.embeddings.position_embeddings.weight
    - mobilebert.embeddings.token_type_embeddings.weight
    trainable: False
    params_strict: True
    start_epoch: 0.0
    end_epoch: 30.0

  - !LearningRateFunctionModifier
    start_epoch: 0.0
    end_epoch: 2.0
    lr_func: linear
    init_lr: 1e-4
    final_lr: 1e-6

  - !LearningRateFunctionModifier
    start_epoch: 2.0
    end_epoch: 28.0
    lr_func: cyclic_linear
    cycle_epochs: 2.0
    init_lr: 1e-4
    final_lr: 5e-5

  - !LearningRateFunctionModifier
    start_epoch: 28.0
    end_epoch: 30.0
    lr_func: linear
    init_lr: 1e-4
    final_lr: 1e-6

pruning_modifiers:
  - !LayerPruningModifier
    start_epoch: 0.0
    end_epoch: 30.0
    layers: [
        'mobilebert.encoder.layer.14',
        'mobilebert.encoder.layer.15',
        'mobilebert.encoder.layer.16',
        'mobilebert.encoder.layer.17',
        'mobilebert.encoder.layer.18',
        'mobilebert.encoder.layer.19',
        'mobilebert.encoder.layer.20',
        'mobilebert.encoder.layer.21',
        'mobilebert.encoder.layer.22',
        'mobilebert.encoder.layer.23',
    ]

  - !OBSPruningModifier
    params: [
      "re:mobilebert.encoder.layer.*.attention.self.query.weight",
      "re:mobilebert.encoder.layer.*.attention.self.key.weight",
      "re:mobilebert.encoder.layer.*.attention.self.value.weight",
      "re:mobilebert.encoder.layer.*.attention.output.dense.weight",
      "re:mobilebert.encoder.layer.*.intermediate.dense.weight",
      "re:mobilebert.encoder.layer.*.output.dense.weight",
      "re:mobilebert.encoder.layer.*.output.bottleneck.dense.weight",
      "re:mobilebert.encoder.layer.*.bottleneck.input.dense.weight",
      "re:mobilebert.encoder.layer.*.bottleneck.attention.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.0.intermediate.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.0.output.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.1.intermediate.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.1.output.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.2.intermediate.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.2.output.dense.weight",
    ]
    init_sparsity: 0.25
    final_sparsity: 0.50
    start_epoch: 2.0
    end_epoch: 28.0
    update_frequency: 2.0
    inter_func: cubic
    global_sparsity: True
    mask_type: block4
    num_grads: 512
    damp: 1e-8
    fisher_block_size: 4
    grad_sampler_kwargs:
      batch_size: 8

distillation_modifiers:
  - !DistillationModifier
    hardness: 1.0
    temperature: 5.5
    distill_output_keys: [start_logits, end_logits]
```

## Step 2: Quantization-Aware Training

Now that we have a 50% semi-structured pruned 14 layers of the MobileBERT model, we apply INT8 quantization-aware training (QAT) on top of it to further improve the performance, while keeping the pruning mask fixed.

Assuming that the SparseML library is installed, the bash script to reproduce our quantization setup is as follows:
```shell
CUDA_VISIBLE_DEVICES=0 python src/sparseml/transformers/question_answering.py \
    --distill_teacher zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base-none \
    --model_name_or_path /path/to/the/pruned/checkpoint/from/the/previous/step \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --optim adamw_torch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --preprocessing_num_workers 8 \
    --seed 42 \
    --num_train_epochs 2 \
    --recipe obert_mobilebert_quantization_recipe.yaml \
    --output_dir my_quantization_output_dir
```

And the *obert_mobilebert_quantization_recipe.yaml* is as follows:
```yaml
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: 2.0

  - !LearningRateFunctionModifier
    start_epoch: 0.0
    end_epoch: 2.0
    lr_func: linear
    init_lr: 3e-5
    final_lr: 0.0

pruning_modifiers:
  - !LayerPruningModifier
    start_epoch: 0.0
    end_epoch: 2.0
    layers: [
        'mobilebert.encoder.layer.14',
        'mobilebert.encoder.layer.15',
        'mobilebert.encoder.layer.16',
        'mobilebert.encoder.layer.17',
        'mobilebert.encoder.layer.18',
        'mobilebert.encoder.layer.19',
        'mobilebert.encoder.layer.20',
        'mobilebert.encoder.layer.21',
        'mobilebert.encoder.layer.22',
        'mobilebert.encoder.layer.23',
    ]

  - !ConstantPruningModifier
    start_epoch: 0.0
    end_epoch: 2.0
    params: [
      "re:mobilebert.encoder.layer.*.attention.self.query.weight",
      "re:mobilebert.encoder.layer.*.attention.self.key.weight",
      "re:mobilebert.encoder.layer.*.attention.self.value.weight",
      "re:mobilebert.encoder.layer.*.attention.output.dense.weight",
      "re:mobilebert.encoder.layer.*.intermediate.dense.weight",
      "re:mobilebert.encoder.layer.*.output.dense.weight",
      "re:mobilebert.encoder.layer.*.output.bottleneck.dense.weight",
      "re:mobilebert.encoder.layer.*.bottleneck.input.dense.weight",
      "re:mobilebert.encoder.layer.*.bottleneck.attention.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.0.intermediate.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.0.output.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.1.intermediate.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.1.output.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.2.intermediate.dense.weight",
      "re:mobilebert.encoder.layer.*.ffn.2.output.dense.weight",
    ]

distillation_modifiers:
  - !DistillationModifier
    hardness: 1.0
    temperature: 5.5
    distill_output_keys: [start_logits, end_logits]

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: 0.0
    disable_quantization_observer_epoch: 1.0
    quantize_embedding_activations: False
    quantize_embeddings: True
    quantize_linear_activations: False
    submodules: ['mobilebert.embeddings', 'mobilebert.encoder', 'qa_outputs']
```

## Final Step: Export to ONNX and Benchmark with DeepSparse
To run the compressed and quantized `obert-mobilebert` model in the DeepSparse engine, we need to export it to ONNX with:
```shell
sparseml.transformers.export_onnx \
    --model_path /path/to/my/compressed/and/quantized/model \
    --task 'question-answering' --sequence_length 384
```

Then benchmark the model in the engine:
```
deepsparse.benchmark /path/to/my/compressed/and/quantized/model.onnx
```

## Additional Info

For more details about our compression approach, please check the Optimal BERT Surgeon (oBERT) paper: [https://arxiv.org/abs/2203.07259](https://arxiv.org/abs/2203.07259).

For the full algorithm implementation, more recipes, examples and tutorials: [https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT).

## Citation info
If you find our models useful, please consider citing our work:
```bibtex
@article{kurtic2022optimal,
  title={The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models},
  author={Kurtic, Eldar and Campos, Daniel and Nguyen, Tuan and Frantar, Elias and Kurtz, Mark and Fineran, Benjamin and Goin, Michael and Alistarh, Dan},
  journal={arXiv preprint arXiv:2203.07259},
  year={2022}
}
```
