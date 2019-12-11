# Beyond DeepJazz

This project focuses on comparison of different generative methods in music generation. We use the exact setting of DeepJazz (preprocessing, grammar, generation pipeline, input / output dimension, certain hyperparameters, etc.) such that a concise and direct of comparison among different models can be observed.

We Acknowledge the great effort by DeepJazz team, it laid a great foundation and interface for our future work. So we can focus on higher-level model selection and comparison.

## Structure

```bash
/data           # store original training data
/result         # store generation result
/utils          # preprocessing, grammar, helper function
/future         # WIP, future work and adaptation
lstm.py         # models, including LSTM, VAE-LSTM, BI-LSTM
generator.py    # project interface
```

## Reference

### Code reference and baseline template

+ Deep Jazz  
https://github.com/jisungk/deepjazz

### Paper reference in implementation

+ Chen, K., Zhang, W., Dubnov, S., Xia, G., & Li, W. (2019, January). The effect of explicit structure encoding of deep neural networks for symbolic music generation. In 2019 International Workshop on Multilayer Music Representation and Processing (MMRP) (pp. 77-84). IEEE.  
https://arxiv.org/pdf/1811.08380.pdf

+ Performance-RNN by Margenta  
https://magenta.tensorflow.org/performance-rnn
  + Dynamics
  + Temperature and randomness
  + Volumes

## Additional Dataset

To download the larger Yamaha e-Piano Competition Dataset, from:
+ Malik, I., & Ek, C. H. (2017). Neural translation of musical style. arXiv preprint arXiv:1708.03535.
Use this link: http://imanmalik.com/assets/dataset/TPD.zip

## Usage

Use `generator.py` for public interface

E.g. use `lstm` model and train 2 epochs

```bash
python generator.py --model-choice "lstm" --epochs 2
```

E.g. use `vae-lstm` model and train 1 epochs

```bash
python generator.py --model-choice "vae-lstm" --epochs 1
```

E.g. use `bi-lstm` model and train 128 epochs

```bash
python generator.py --model-choice "bi-lstm" --epochs 128
```

## Requirement

```bash
python 2.7
keras
tensorflow
music21
numpy
```
