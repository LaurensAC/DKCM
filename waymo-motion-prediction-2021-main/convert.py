import argparse
import os

import numpy as np
import timm
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self, model_name="xception71", in_channels=47, time_limit=80, n_traj=8
    ):
        super().__init__()

        self.n_traj = n_traj
        self.time_limit = time_limit
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes=self.n_traj * 2 * self.time_limit + self.n_traj,
        )


    def forward(self, x):
        outputs = self.model(x)

        confidences_logits, logits = (
            outputs[:, : self.n_traj],
            outputs[:, self.n_traj :],
        )
        logits = logits.view(-1, self.n_traj, self.time_limit, 2)

        return confidences_logits, logits


weights_path=".\\model\\base_model.pth"

model = Model()
checkpoint = torch.load(weights_path)['model_state_dict']
#model.load_state_dict(checkpoint)
for key in list(checkpoint.keys()):
    checkpoint['model.'+key] = checkpoint[key]
    del checkpoint[key]
model.load_state_dict(checkpoint)
traced_script_module = torch.jit.trace(
                        model,
                        torch.rand(
                            1, 47, 224, 224
                        ),
                    )
traced_script_module.save("base_model.pt")
