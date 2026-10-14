import torch
import model
import torch.optim as optim
import transformers
from torch_textdataset import TextDataset
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader




# Creating your data loader
bin_path = "datasets/wikiset.bin" ### Insert here your data.bin already tokenized, or use wikiset.bin (see README)
txt_set = TextDataset(bin_path, 256)
dataload = DataLoader(txt_set, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)

# Hyperparameters of your model to define
vocab_size = 16384
d_model = 512
max_seq_len = 256
n_heads = 8
n_layers = 6

config_json = {
    "vocab_size" : vocab_size,
    "d_model" : d_model,
    "max_seq_len" : max_seq_len,
    "n_heads" : n_heads,
    "n_layers" : n_layers, 
}

with open("model_config.json", "w", encoding="utf-8") as f:
    json.dump(config_json, f, indent=4)

config_dict_path = "model_config.json"

# Instantiating model  
zachGPT = model.Transformer(vocab_size, d_model, max_seq_len, n_heads, n_layers)
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)
zachGPT.to(device)


# Instanciating AdamW optimizer and its hyperparameters
lr = 0.0005
weight_decay = 0.01
betas = (0.9, 0.95)
eps = 1e-8
optimizer = optim.AdamW(
    zachGPT.parameters(),
    lr=lr,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
)
# Training configuration
target_steps = 50000
total_steps = min(target_steps, len(dataload))
lr_scheduler = transformers.get_scheduler("cosine", optimizer, num_warmup_steps=500, num_training_steps=total_steps) # The learning rate function with warmup and cosine decay towards 0

global_step = 0
zachGPT.train()

# Parameters for pyplot
steps_history = []
loss_history = []

running_loss = 0.0
log_interval = 100

for x, y in dataload:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = zachGPT(x)
            loss = model.compute_loss(logits, y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(zachGPT.parameters(), max_norm=1.0)

    optimizer.step()
    lr_scheduler.step()

    running_loss += loss.item()

    if global_step % log_interval == 0 and global_step > 0:
        avg_loss = running_loss / log_interval
        steps_history.append(global_step)
        loss_history.append(avg_loss)
        running_loss = 0.0

        if global_step % 500 == 0:
            print(
                f"Step {global_step:5d}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}"
            )

    if global_step % 2000 == 0 and global_step > 0:
        checkpoint = {
            "model_state_dict": zachGPT.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "global_step": global_step,
            "total_steps": total_steps,
            "loss": loss.item(),
            "model_config": config_json,
            "steps_history": steps_history,
            "loss_history": loss_history,
        }
        torch.save(checkpoint, f"checkpoint_step_{global_step}.pt")
    if global_step % 500 == 0 and global_step > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(
            steps_history,
            loss_history,
            label="Training Loss",
            color="#2563eb",
            linewidth=2,
        )
        plt.title(
            f"zachGPT Pre-training Loss Curve (Step {global_step}/{total_steps})"
        )
        plt.xlabel("Optimization Steps")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("assets/loss_curve.png", dpi=150)
        plt.close()  

    global_step += 1
    if global_step >= total_steps:
        break

plt.figure(figsize=(10, 5))
plt.plot(
    steps_history,
    loss_history,
    label="Final Training Loss",
    color="#16a34a",
    linewidth=2,
)
plt.title("zachGPT Pre-training Loss Curve - Final")
plt.xlabel("Optimization Steps")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("assets/loss_curve.png", dpi=300)
plt.close()


checkpoint = {
    "model_state_dict": zachGPT.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": lr_scheduler.state_dict(),
    "global_step": global_step,
    "total_steps": total_steps,
    "loss": loss.item(),
    "model_config": config_json,
    "steps_history": steps_history,
    "loss_history": loss_history,
}

torch.save(checkpoint, f"zachGPT_pretrained_{total_steps}.pt")