import torch, os
import matplotlib.pyplot as plt
from tqdm import tqdm

layers = range(32)
types = ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"]
indices = range(0, 128, 8)
os.makedirs("tempfig2", exist_ok=True)

for layer in tqdm(layers):
    for module_type in types:
        for index in indices:
            fname = f"Layer{layer}_{module_type}_inp{index}"
            input = torch.load(f"Tensors/{fname}.pt")
            stds = torch.std(input=input, dim=1)
            means = torch.mean(input=input, dim=1)
            cvs = stds / means
            plt.hist(cvs.detach().cpu(), bins=150)
            plt.title(f"Layer {layer} | {module_type} | {index}th input")
            plt.savefig(f"tempfig2/{fname}.png")
            plt.clf()