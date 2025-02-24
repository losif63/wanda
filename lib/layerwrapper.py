import torch
import torch.nn as nn
import os

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name
        self.input_avg = torch.zeros((self.columns), device=self.dev)

    # self.scaler_row contains the processed input!
    # The input is L2 normed across the batch * sequence_length dimension --> average
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # os.makedirs(os.path.dirname(f"Tensors/Layer{self.i}_{self.name}_inp{self.nsamples}.pt"), exist_ok=True)
        # torch.save(inp, f"Tensors/Layer{self.i}_{self.name}_inp{self.nsamples}.pt")  

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.input_avg *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        self.input_avg += torch.mean(inp, dim=1) / self.nsamples