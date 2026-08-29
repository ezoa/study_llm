# GPT_CONFIG_124M = {
# "vocab_size": 50257,  # Vocabulary size
# "context_length": 1024,      # Context length
# "emb_dim": 768, # Embedding dimension
# "n_heads": 12, # Number of attention heads
# "n_layers": 12,# Number of layers
# "drop_rate": 0.1, # Dropout rate
# "qkv_bias": False # Query-Key-Value bias
# }
# import torch
# import torch.nn as nn
# class DummyGPTModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
#         self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
#         self.drop_emb = nn.Dropout(cfg["drop_rate"])
#         self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]) #A
#         self.final_norm = DummyLayerNorm(cfg["emb_dim"]) #B
#         self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False
#         )
#     def forward(self, in_idx):
#         batch_size, seq_len = in_idx.shape
#         tok_embeds = self.tok_emb(in_idx)
#         pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
#         x = tok_embeds + pos_embeds
#         x = self.drop_emb(x)
#         x = self.trf_blocks(x)
#         x = self.final_norm(x)
#         logits = self.out_head(x)
#         return logits
# class DummyTransformerBlock(nn.Module): #C
#     def __init__(self, cfg):
#         super().__init__()
#     def forward(self, x): #D
#         return x
# class DummyLayerNorm(nn.Module): #E
#     def __init__(self, normalized_shape, eps=1e-5): #F
#         super().__init__()
#     def forward(self, x):
#         return x
# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)
# print(batch)

# torch.manual_seed(123)
# model = DummyGPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)

# torch.manual_seed(123)
# batch_example = torch.randn(2, 5) #A
# layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# out = layer(batch_example)
# print(out)

# mean = out.mean(dim=-1, keepdim=True)
# var = out.var(dim=-1, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# out_norm = (out - mean) / torch.sqrt(var)
# mean = out_norm.mean(dim=-1, keepdim=True)
# var = out_norm.var(dim=-1, keepdim=True)
# print("Normalized layer outputs:\n", out_norm)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# class LayerNorm(nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()
#         self.eps = 1e-5
#         self.scale = nn.Parameter(torch.ones(emb_dim))
#         self.shift = nn.Parameter(torch.zeros(emb_dim))
#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, keepdim=True, unbiased=False)
#         norm_x = (x - mean) / torch.sqrt(var + self.eps)
#         return self.scale * norm_x + self.shift
# ln = LayerNorm(emb_dim=5)
# out_ln = ln(batch_example)
# mean = out_ln.mean(dim=-1, keepdim=True)
# var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)


# class GELU(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(
#             torch.sqrt(torch.tensor(2.0 / torch.pi)) *
#             (x + 0.044715 * torch.pow(x, 3))
#         ))
# import matplotlib.pyplot as plt
# gelu, relu = GELU(), nn.ReLU()
# x = torch.linspace(-3, 3, 100) #A
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8, 3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU",
# "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
# plt.tight_layout()
# # plt.show()

# class FeedForward(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
#             GELU(),
#             nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
#     def forward(self, x):
#         return self.layers(x)
# ffn = FeedForward(GPT_CONFIG_124M)
# x = torch.rand(2, 3, 768) #A
# out = ffn(x)
# print(out.shape)


# torch.Size([2, 3, 768])

# class ExampleDeepNeuralNetwork(nn.Module):
#     def __init__(self, layer_sizes, use_shortcut):
#         super().__init__()
#         self.use_shortcut = use_shortcut
#         self.layers = nn.ModuleList([
#             # Implement 5 layers
#             nn.Sequential(nn.Linear(layer_sizes[0],layer_sizes[1]), GELU()),
#             nn.Sequential(nn.Linear(layer_sizes[1],layer_sizes[2]), GELU()),
#             nn.Sequential(nn.Linear(layer_sizes[2],layer_sizes[3]), GELU()),
#             nn.Sequential(nn.Linear(layer_sizes[3],layer_sizes[4]), GELU()),
#             nn.Sequential(nn.Linear(layer_sizes[4],layer_sizes[5]), GELU())])
#     def forward(self, x):
#         for layer in self.layers:
#             # Compute the output of the current layer
#             layer_output = layer(x)
#             # Check if shortcut can be applied
#             if self.use_shortcut and x.shape ==layer_output.shape:

#                 x = x + layer_output
#             else:
#                 x = layer_output
#             return x

# layer_sizes = [3, 3, 3, 3, 3, 1]
# sample_input = torch.tensor([[1., 0., -1.]])
# torch.manual_seed(123) # specify random seed for the initialweights for reproducibility
# model_without_shortcut = ExampleDeepNeuralNetwork(
#     layer_sizes, use_shortcut=False
# )

# def print_gradients(model, x):
#     # Forward pass
#     output = model(x)
#     target = torch.tensor([[0.]])
#     # Calculate loss based on how close the target
#     # and output are
#     loss = nn.MSELoss()
#     loss = loss(output, target)
#     # Backward pass to calculate the gradients
#     loss.backward()
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             # Print the mean absolute gradient of the weights
#             print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
# print_gradients(model_without_shortcut, sample_input)
# # def print_gradients(model, x):
# #     # Clear previous gradients
# #     model.zero_grad()

# #     # Forward pass
# #     output = model(x)

# #     # Target must have the same shape as output
# #     target = torch.zeros_like(output)

# #     # Calculate loss
# #     loss = nn.MSELoss()(output, target)

# #     # Backward pass
# #     loss.backward()

# #     # Print gradients
# #     for name, param in model.named_parameters():
# #         if 'weight' in name and param.grad is not None:
# #             print(
# #                 f"{name} has gradient mean of "
# #                 f"{param.grad.abs().mean().item()}"
# #             )
# print_gradients(model_without_shortcut, sample_input)
# torch.manual_seed(123)
# model_with_shortcut = ExampleDeepNeuralNetwork(
#     layer_sizes, use_shortcut=True
# )
# print_gradients(model_with_shortcut, sample_input)
 

import torch

import torch.nn.functional as F


from torch.autograd import grad

y =  torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = w1 * x1 + b

a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a,y)

# grad_L_w1 = grad(loss, w1, retain_graph=True)
# grad_L_b = grad(loss, b, retain_graph=True)

# print("Gradient of loss w.r.t w1:", grad_L_w1)
# print("Gradient of loss w.r.t b:", grad_L_b)

loss.backward ()
print("Gradient of loss w.r.t w1 (using backward()):", w1.grad)
print("Gradient of loss w.r.t b (using backward()):", b.grad)


# A.5 Implementing multilayer neural networks

class NeutralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        torch.nn.
        self.layers = torch.nn.Sequential(
            # 1st hidden layer 
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # Output layer
            torch.nn.Linear(20, num_outputs)

        )
    def forward(self, x):
        logits = self.layers(x)
        return logits



model = NeutralNetwork(num_inputs=50, num_outputs=3)
print(model)

