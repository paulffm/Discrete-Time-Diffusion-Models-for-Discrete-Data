# Discrete-Time Diffusion Models for Discrete Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/paulffm/Discrete-Time-Diffusion-Models-for-Discrete-Data/blob/main/LICENSE)

Unofficial **PyTorch** reimplementations of the
papers [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/pdf/2107.03006) (D3PM)
by J. Austin et al. and [Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://arxiv.org/abs/2102.05379)
by E. Hoogeboom et al.

<p align="center">
  <img src="assets/forward_reverse_process.png"  alt="1" width = 820px height = 250px >
</p>

## Installation

Follow these steps to clone the repository and install the dependencies:

### 1. Clone the repository

Clone the repository using the following command:

```sh
git clone https://github.com/paulffm/Discrete-Time-Diffusion-Models-for-Discrete-Data.git
cd Discrete-Time-Diffusion-Models-for-Discrete-Data
```

### 2. Create a virtual environment

Create a virtual environment to install dependencies in isolation:

```sh
python -m venv myvenv
source myvenv/bin/activate  # On Windows use `myvenv\Scripts\activate`
```

### 3. Install dependencies

Install the necessary dependencies using pip:

```sh
pip install -r requirements.txt
```

## Usage

This implementation provides an example script **train_d3pm.py** for training D3PM models to generate [MNIST](http://yann.lecun.com/exdb/mnist/) or [maze](https://github.com/Turidus/Python-Maze/tree/master). In this script you can simply use my provided configs and start training or retraining your models. You just need to set the correct paths in the beginning of the script, i.e.:

```python
def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_location = os.path.join(script_dir, f"SavedModels/MNIST/")
    save_location_png = os.path.join(save_location, "PNGs/")
    dataset_location = os.path.join(script_dir, 'lib/datasets')

    train_resume = False
    print(save_location)
    if not train_resume:
        cfg = get_config()
        bookkeeping.save_config(cfg, save_location)

    else:
        model_name = "model_name.pt"
        date = "2024-05-10"
        config_name = "config_001.yaml"
        config_path = os.path.join(save_location, date, config_name)
        cfg = bookkeeping.load_config(config_path)
```
In addition, you need to set a location in the config files where you want to save you trained models:
```python
save_directory = "SavedModels/MNIST/"
```
## Note

Infos to the maze dataset can be found [here](https://github.com/Turidus/Python-Maze/tree/master).

## Results

According to [PaperwithCode](https://paperswithcode.com/sota/image-generation-on-mnist), the [D3PM](https://arxiv.org/pdf/2107.03006) framework achieves state-of-the-art performance in terms of FID score on the MNIST dataset, surpassing previous state-of-the-art models. However, it lacks behind the [tauLDR](https://arxiv.org/pdf/2205.14987) framework, combined with negative log-likelihood loss and the Midpoint Tau-Leaping sampler from my [repository](https://github.com/paulffm/Continuous-Time-Diffusion-Models-for-Discrete-Data):

| Rank | Model | FID |
| ---- | ----- | --- |
| 1    | [tauLDR](https://arxiv.org/pdf/2205.14987) + $L_{\text{ll}}$ + Midpoint Tau-Leaping | 1.75 |
| 2    | [tauLDR](https://arxiv.org/pdf/2205.14987) + $L_{\text{CTEll}}$+ Midpoint Tau-Leaping | 2.40 |
| 2    | [D3PM](https://arxiv.org/pdf/2107.03006) + \(L_{\text{DTEll}}\) | 1.88 |
| 3    | [Sliced Iterative Normalizing Flows](https://arxiv.org/pdf/2007.00674v3) | 4.5 |
| 4    | [Generative Latent Flow + perceptual loss](https://arxiv.org/pdf/1905.10485v2) | 5.8 |
| 5    | [HypGan](https://arxiv.org/pdf/2102.05567v1) | 7.87 |

In the above table:

- $L_{\text{ll}}$ represents the negative log-likelihood loss.
- $L_{\text{CTEll}} = L_\text{cvb} + \lambda L_{\text{ll}}$ denotes a combination of the continuous-time ELBO and negative log-likelihood loss.
- $L_{\text{CTEll}} = L_\text{vb} + \lambda L_{\text{ll}}$ denotes a combination of the discrete-time ELBO and negative log-likelihood loss.
  In both cases, $\lambda = 0.001$.

Some generated MNIST and maze samples:

<p align="center">
  <img src="assets/mnist_samples.png" alt="Image 1" width="45%">
  <img src="assets/maze_samples.png" alt="Image 2" width="45%">
</p>

## Reference

```bibtex
@article{austin2021structured,
  title={Structured denoising diffusion models in discrete state-spaces},
  author={Austin, Jacob and Johnson, Daniel D and Ho, Jonathan and Tarlow, Daniel and Van Den Berg, Rianne},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={17981--17993},
  year={2021}
}
@article{hoogeboom2021argmax,
  title={Argmax flows and multinomial diffusion: Learning categorical distributions},
  author={Hoogeboom, Emiel and Nielsen, Didrik and Jaini, Priyank and Forr{\'e}, Patrick and Welling, Max},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={12454--12465},
  year={2021}
}
```
