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

This implementation provides an example script **train_d3pm.py** for training D3PM models to generate [MNIST](http://yann.lecun.com/exdb/mnist/) or [maze](https://github.com/Turidus/Python-Maze/tree/master) data. In this script you can simply use my provided configs and start training or continue training your models. You just need to set the correct paths in the beginning of the script, i.e.:

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
        date = "2024-06-10"
        config_name = "config_001.yaml"
        config_path = os.path.join(save_location, date, config_name)
        cfg = bookkeeping.load_config(config_path)
```

The configuration files (`TAUnSDDM/config`) are provided to simplify the training and sampling process. A configuration file tailored for generating MNIST data with a U-Net includes the following parameters:

| Parameter                 | Description                                          | Type   |
|---------------------------|------------------------------------------------------|--------|
| device                    | Device to be used for training                       | str    |
| distributed               | Whether to use distributed training                  | bool   |
| num_gpus                  | Number of GPUs to use                                | int    |
| training.train_step_name  | Name of the training step                            | str    |
| training.n_iters          | Number of training iterations                         | int    |
| training.clip_grad        | Whether to clip gradients                            | bool   |
| training.grad_norm        | Value to normalize gradients                          | int    |
| training.warmup           | Number of warmup steps                               | int    |
| data.name                 | Name of the dataset                                  | str    |
| data.train                | Whether to use training data                         | bool   |
| data.download             | Whether to download dataset                          | bool   |
| data.S                    | Number of discrete states                            | int    |
| data.batch_size           | Batch size                                           | int    |
| data.shuffle              | Whether to shuffle data                              | bool   |
| data.image_size           | Size of the input image                              | int    |
| data.random_flips         | Whether to apply random flips to the data            | bool   |
| data.use_augm             | Whether to use data augmentation                     | bool   |
| model.name                | Name of the model                                    | str    |
| model.padding             | Whether to use padding in the model                  | bool   |
| model.ema_decay           | Exponential moving average decay rate                | float  |
| model.ch                  | Number of channels in the model                      | int    |
| model.num_res_blocks      | Number of residual blocks in the model               | int    |
| model.ch_mult             | Multiplier for channel dimensions in the model       | list   |
| model.input_channels      | Number of input channels                             | int    |
| model.scale_count_to_put_attn | Scaling factor for attention resolution          | int    |
| model.data_min_max        | Minimum and maximum data values                      | list   |
| model.dropout             | Dropout rate in the model                            | float  |
| model.skip_rescale        | Whether to rescale skipped connections               | bool   |
| model.time_embed_dim      | Dimension of time embedding                          | int    |
| model.time_scale_factor   | Scaling factor for time dimension                    | int    |
| model.fix_logistic        | Whether to fix logistic outputs                      | bool   |
| model.model_output        | Type of model output                                 | str    |
| model.num_heads           | Number of attention heads                            | int    |
| model.attn_resolutions    | Resolutions for attention mechanisms                 | list   |
| model.concat_dim          | Dimension for concatenation in the model             | int    |
| model.type                | Type of diffusion model                              | str    |
| model.start               | Start value for noise schedule of diffusion model                      | float  |
| model.stop                | Stop value for noise schedule of for diffusion model                       | float  |
| model.num_timesteps       | Number of time steps for diffusion model             | int    |
| model.model_prediction    | Type of model prediction                             | str    |
| model.transition_mat_type | Type of transition matrix                           | str    |
| model.loss_type           | Type of loss function                                | str    |
| model.hybrid_coeff        | Coefficient for hybrid loss function                 | float  |
| optimizer.name            | Name of the optimizer                                | str    |
| optimizer.lr              | Learning rate of the optimizer                       | float  |
| saving.sample_plot_path   | Path to save sample plots                            | str    |
| saving.checkpoint_freq    | Frequency of saving model checkpoints                | int    |

## Results

According to [PaperwithCode](https://paperswithcode.com/sota/image-generation-on-mnist), the [D3PM](https://arxiv.org/pdf/2107.03006) framework achieves state-of-the-art performance in terms of FID score on the MNIST dataset, surpassing previous state-of-the-art models. However, it lacks behind the [tauLDR](https://arxiv.org/pdf/2205.14987) framework, combined with negative log-likelihood loss and the Midpoint Tau-Leaping sampler from my [repository](https://github.com/paulffm/Continuous-Time-Diffusion-Models-for-Discrete-Data):

| Rank | Model | FID |
| ---- | ----- | --- |
| 1    | [tauLDR](https://arxiv.org/pdf/2205.14987) + $L_{\text{ll}}$ + Midpoint Tau-Leaping | 1.75 |
| 2    | [D3PM](https://arxiv.org/pdf/2107.03006) + $L_{\text{DTEll}}$ | 1.88 |
| 3    | [tauLDR](https://arxiv.org/pdf/2205.14987) + $L_{\text{CTEll}}$+ Midpoint Tau-Leaping | 2.40 |
| 4    | [Sliced Iterative Normalizing Flows](https://arxiv.org/pdf/2007.00674v3) | 4.5 |
| 5    | [Generative Latent Flow + perceptual loss](https://arxiv.org/pdf/1905.10485v2) | 5.8 |
| 6    | [HypGan](https://arxiv.org/pdf/2102.05567v1) | 7.87 |

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

## Note

Infos to the maze dataset can be found [here](https://github.com/Turidus/Python-Maze/tree/master).

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
