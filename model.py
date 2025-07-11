from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
    

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        # sample random timestep
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)

        # add noise and get xt
        xt, noise = self.var_scheduler.add_noise(x0, timestep, noise)

        # Predict the noise using the network
        if class_label is not None:
            # Conditional generation
            noise_pred = self.network(xt, timestep, class_label=class_label)
        else:
            # Unconditional generation
            noise_pred = self.network(xt, timestep)

        # mse loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
            self,
            batch_size,
            return_traj=False,
            class_label: Optional[torch.Tensor] = None,
            guidance_scale: Optional[float] = 1.0,
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            raise NotImplementedError("ignore")

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]

            # to train on gpu
            t_on_device = t.to(self.device)

            noise_pred = self.network(x_t, timestep=t_on_device)

            x_t_prev = self.var_scheduler.step(x_t, t_on_device, noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
