# ResNet-50: Alternating Compressed/DeCompressed Training (AC/DC) applied to the ResNet model on ImageNet

[AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks](https://arxiv.org/abs/2106.12379) outperforms existing sparse training methods in accuracy at similar computational budgets; at high sparsity levels, AC/DC even outperforms existing methods that rely on accurate pre-trained dense models. An important property of AC/DC is that it allows co-training of dense and sparse models, yielding accurate sparse-dense model pairs at the end of the training process. This is useful in practice, where compressed variants may be desirable for deployment in resource-constrained settings without re-doing the entire training flow, and also provides us with insights into the accuracy gap between dense and compressed models.

This recipe defines the hyperparams necessary to prune a ResNet-50 model to 85% and quantize it on an image classification task for the [ImageNet 2012 dataset](https://image-net.org/challenges/LSVRC/2012/).
To vary hyperparams either edit the recipe or supply the --recipe_args argument to the training commands.
For example, the following appended to the training commands will change the number of epochs:
```bash
--recipe_args '{"num_epochs":150}'
```

## Training

To set up the training environment, [install SparseML with PyTorch](https://github.com/neuralmagic/sparseml#installation)

The following command is used to prune and quantize a ResNet-50 model on the ImageNet dataset.

```bash
sparseml.image_classification.train \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned85_quant-none-vnni?recipe_type=original \
    --pretrained True \
    --arch-key resnet50 \
    --dataset imagenet \
    --dataset-path /PATH/TO/IMAGENET \
    --train-batch-size 256 \
    --test-batch-size 1000 \
    --loader-num-workers 16 \
    --model-tag resnet50-imagenet-pruned85_quant-none-vnni
```

## Evaluation

This model achieves 75.6% top1 accuracy on the validation set. The following command can be used to verify accuracy.

```bash
sparseml.image_classification.train \
    --eval-mode \
    --checkpoint-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned85_quant-none-vnni \
    --arch-key resnet50 \
    --dataset imagenet \
    --dataset-path /PATH/TO/IMAGENET \
    --train-batch-size 256 \
    --test-batch-size 256 \
    --loader-num-workers 16 \
    --model-tag resnet50-imagenet-pruned85_quant-none-vnni-eval
```

# Recipe

```yaml
version: 1.1.0

# General Variables
num_epochs: 206
lr_warmup_epochs: 5
init_lr: 0.0512
warmup_lr: 0.256
weight_decay: 0.00001

# Quantization variables
quantization_epochs: 6
quantization_start_epoch: eval(num_epochs - quantization_epochs)
quantization_end_epoch: eval(num_epochs)
quantization_init_lr: 0.00005
quantization_observer_epochs: 1
quantization_keep_bn_epochs: 2

# Pruning Variables
pruning_epochs_fraction: 0.925
pruning_epochs: eval(int((num_epochs - quantization_epochs) * pruning_epochs_fraction))
pruning_start_epoch: eval(lr_warmup_epochs)
pruning_end_epoch: eval(pruning_start_epoch + pruning_epochs)
pruning_update_frequency: 5
pruning_sparsity: 0.85
pruning_final_lr: 0.0


training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(lr_warmup_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(warmup_lr)

  - !LearningRateFunctionModifier
    start_epoch: eval(lr_warmup_epochs)
    end_epoch: eval(quantization_start_epoch)
    lr_func: cosine
    init_lr: eval(warmup_lr)
    final_lr: eval(pruning_final_lr)

  - !SetWeightDecayModifier
    start_epoch: 0
    end_epoch: eval(quantization_start_epoch)
    weight_decay: eval(weight_decay)

pruning_modifiers:
  - !ACDCPruningModifier
    compression_sparsity: eval(pruning_sparsity)
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)
    mask_type: block4
    params: 
      - sections.0.0.conv1.weight
      - sections.0.0.conv3.weight
      - sections.0.0.identity.conv.weight
      - sections.0.1.conv3.weight
      - sections.0.2.conv3.weight
      - sections.1.0.conv3.weight
      - sections.0.0.conv2.weight
      - sections.0.1.conv1.weight
      - sections.0.2.conv1.weight
      - sections.1.0.conv1.weight
      - sections.1.0.identity.conv.weight
      - sections.1.1.conv3.weight
      - sections.1.2.conv3.weight
      - sections.1.3.conv3.weight
      - sections.2.0.conv3.weight
      - sections.0.1.conv2.weight
      - sections.0.2.conv2.weight
      - sections.1.0.conv2.weight
      - sections.1.1.conv1.weight
      - sections.1.2.conv1.weight
      - sections.1.3.conv1.weight
      - sections.2.0.conv1.weight
      - sections.2.0.identity.conv.weight
      - sections.2.1.conv3.weight
      - sections.2.2.conv3.weight
      - sections.2.3.conv3.weight
      - sections.2.4.conv3.weight
      - sections.2.5.conv3.weight
      - sections.3.0.conv3.weight
      - sections.3.1.conv3.weight
      - sections.3.2.conv3.weight
      - sections.1.1.conv2.weight
      - sections.1.2.conv2.weight
      - sections.1.3.conv2.weight
      - sections.2.0.conv2.weight
      - sections.2.1.conv1.weight
      - sections.2.2.conv1.weight
      - sections.2.3.conv1.weight
      - sections.2.4.conv1.weight
      - sections.2.5.conv1.weight
      - sections.3.0.conv1.weight
      - sections.3.0.identity.conv.weight
      - sections.3.1.conv1.weight
      - sections.3.2.conv1.weight
      - sections.1.1.conv2.weight
      - sections.1.2.conv2.weight
      - sections.1.3.conv2.weight
      - sections.2.0.conv2.weight
      - sections.2.1.conv1.weight
      - sections.2.2.conv1.weight
      - sections.2.3.conv1.weight
      - sections.2.4.conv1.weight
      - sections.2.5.conv1.weight
      - sections.3.0.conv1.weight
      - sections.3.0.identity.conv.weight
      - sections.3.1.conv1.weight
      - sections.3.2.conv1.weight
      - sections.2.1.conv2.weight
      - sections.2.2.conv2.weight
      - sections.2.3.conv2.weight
      - sections.2.4.conv2.weight
      - sections.2.5.conv2.weight
      - sections.3.0.conv2.weight
      - sections.3.1.conv2.weight
      - sections.3.2.conv2.weight
    global_sparsity: False

quantization_modifiers:
  - !SetLearningRateModifier
    start_epoch: eval(quantization_start_epoch)
    learning_rate: eval(quantization_init_lr)

  - !LearningRateFunctionModifier
    final_lr: 0.0
    init_lr: eval(quantization_init_lr)
    lr_func: cosine
    start_epoch: eval(quantization_start_epoch + quantization_keep_bn_epochs)
    end_epoch: eval(num_epochs)

  - !QuantizationModifier
    start_epoch: eval(quantization_start_epoch)
    submodules:
      - input
      - sections
    disable_quantization_observer_epoch: eval(quantization_start_epoch + quantization_observer_epochs)
    freeze_bn_stats_epoch: eval(quantization_start_epoch + quantization_keep_bn_epochs)
```