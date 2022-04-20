import numpy as np
import torch
from torch import nn

class VelEstimator(nn.Module):
    def __init__(self,num_input) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_input,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )
        print(f"Velocity Estimator MLP: {self.linear_relu_stack}")


    def forward(self,x):
        return self.linear_relu_stack(x)

    def eval_inference(self, input_e): #使用estimator网络推断body vel
        vel = self.linear_relu_stack(input_e)
        return vel

    
