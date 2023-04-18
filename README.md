# SegFormer-tf

This repository is about an implementation of the research paper "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"

SegFormer is a Transformer-based framework for semantic segmentation that unifies Transformers with lightweight multilayer perceptron (MLP) decoders.

#### Model Architecture :

<p align="center">
  <img src="https://github.com/IMvision12/SegFormer-tf/blob/main/images/arch.png" title="graph">
</p>

#### Detailed overview of MiT :

<p align="center">
  <img src="https://github.com/IMvision12/SegFormer-tf/blob/main/images/mit.png" title="graph">
</p>

# Usage:

Clone Github Repo: 

```bash
$ git clone https://github.com/IMvision12/SegFormer-tf
$ cd SegFormer-tf
```

Then import model

```py
import tensorflow as tf
from models import SegFormer_B3
model = SegFormer_B3(input_shape = (224, 224, 3), num_classes = 19)
print(model.summary())
```


# References

[1] SegFormer paper: https://arxiv.org/pdf/2105.15203

[2] Official SegFormer Repo: https://github.com/NVlabs/SegFormer
