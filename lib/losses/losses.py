import torch
import torch.nn as nn
import lib.losses.losses_utils as losses_utils
import torch.autograd.profiler as profiler
import torch.nn.functional as F
import lib.utils.utils as utils
import time
from lib.models.model_utils import get_logprob_with_logits
from lib.d3pm import make_diffusion

class d3pm_loss:
    def __init__(self, cfg, diffusion):
        self.cfg = cfg
        self.diffusion = diffusion
        self.device = cfg.device
        self.num_timesteps = cfg.model.num_timesteps

    def calc_loss(self, minibatch, state):
        """

        Args:
            minibatch (_type_): _description_
            state (_type_): _description_
            writer (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        model = state["model"]
        # t = np.random.randint(size=(img.shape[0],), low=0, high=self.num_timesteps, dtype=np.int32)
        t = (torch.randint(low=0, high=(self.num_timesteps), size=(minibatch.shape[0],))).to(self.device)
        loss = self.diffusion.training_losses(model, minibatch, t).mean()
        return loss
