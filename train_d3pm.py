import torch
import lib.utils.bookkeeping as bookkeeping
from tqdm import tqdm
import matplotlib.pyplot as plt
import ssl
import os
from lib.d3pm import make_diffusion
from config.mnist_config.config_mnist_d3pm import get_config
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.mnist as mnist
import lib.datasets.dataset_utils as dataset_utils
import lib.losses.losses as losses
import lib.losses.losses_utils as losses_utils
import lib.training.training as training
import lib.training.training_utils as training_utils
import lib.optimizers.optimizers as optimizers
import lib.optimizers.optimizers_utils as optimizers_utils
import time
from torch.utils.data import DataLoader
import numpy as np
from ruamel.yaml.scalarfloat import ScalarFloat
import time
from lib.losses.losses import d3pm_loss

# MLE: Iter: 300000 168214.7792992592


def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_location = os.path.join(script_dir, f"SavedModels/MNIST/")
    save_location_png = os.path.join(save_location, "PNGs/")
    # dataset_location = os.path.join(script_dir, 'lib/datasets')
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

    device = torch.device(cfg.device)

    model = model_utils.create_model(cfg, device)

    optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)

    state = {"model": model, "optimizer": optimizer, "n_iter": 0}

    if train_resume:
        checkpoint_path = os.path.join(save_location, date, model_name)
        state = bookkeeping.load_state(state, checkpoint_path, device)
        cfg.training.n_iters = 300000
        cfg.sampler.sample_freq = 500000000000
        cfg.saving.checkpoint_freq = 5000
        cfg.sampler.num_steps = 1000
        cfg.sampler.corrector_entry_time = ScalarFloat(0.0)
        # bookkeeping.save_config(cfg, save_location)


    diffusion = make_diffusion(cfg.model)

    loss = d3pm_loss(cfg, diffusion)

    training_step = training_utils.get_train_step(cfg)

    dataset = dataset_utils.get_dataset(cfg, device, dataset_location)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle)

    print("Info:")
    print("--------------------------------")
    print("State Iter:", state["n_iter"])
    print("--------------------------------")
    print("Name Dataset:", cfg.data.name)
    print("Loss Name:", cfg.loss.name)
    print("--------------------------------")
    print("Model Name:", cfg.model.name)
    print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))
    print("Sampler:", cfg.sampler.name)

    n_samples = 16
    training_loss = []
    exit_flag = False
    n = 1
    start = time.time()
    num_timesteps = cfg.model.num_timesteps
    print("Num steps", num_timesteps)
    while True:
        for minibatch in tqdm(dataloader):
            minibatch = minibatch.to(device)
            l = training_step.step(state, minibatch.long(), loss)

            training_loss.append(l.item())

            if (state["n_iter"] + 1) % cfg.saving.checkpoint_freq == 0 or state[
                "n_iter"
            ] == cfg.training.n_iters - 1:
                bookkeeping.save_state(state, save_location)
                print("Model saved in Iteration:", state["n_iter"] + 1)
                saving_train_path = os.path.join(
                    save_location_png, f"loss_{cfg.loss.name}{state['n_iter']}.png"
                )
                saving_train_loss = os.path.join(
                    save_location_png, f"loss_{cfg.loss.name}{state['n_iter']}.npy"
                )
                plt.plot(training_loss)
                np.save(saving_train_loss, training_loss)
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.title("Training Loss")
                plt.savefig(saving_train_path)
                plt.close()

            state["n_iter"] += 1
            if state["n_iter"] > cfg.training.n_iters - 1:
                exit_flag = True
                break

        if exit_flag:
            break

    saving_train_path = os.path.join(
        save_location_png, f"loss_{cfg.loss.name}{state['n_iter']}.png"
    )
    plt.plot(training_loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(saving_train_path)
    plt.close()


if __name__ == "__main__":
    main()
