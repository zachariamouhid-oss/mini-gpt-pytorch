import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import transformers
from model import Transformer
from torch_textdataset import TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resume_training(
    checkpoint_path: str,
    data_file_path: str,
    target_total_steps: int | None = None,
):
    os.makedirs("assets", exist_ok=True)

    # Checkpoint loading
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    # Rebuilding model and training parameters

    model_config = checkpoint["model_config"]
    zachGPT = Transformer(**model_config)
    zachGPT.load_state_dict(checkpoint["model_state_dict"])
    zachGPT.to(device)
    zachGPT.train()

    global_step = checkpoint["global_step"]
    total_steps = (
        target_total_steps
        if target_total_steps is not None
        else checkpoint["total_steps"]
    )

    if global_step >= total_steps:
        print(
            f"Attention: global_step ({global_step}) >= total_steps ({total_steps}). "
            "Renseigne un target_total_steps supérieur pour continuer l'entraînement."
        )
        return

    steps_history = checkpoint.get("steps_history", [])
    loss_history = checkpoint.get("loss_history", [])

    # Setting back op^timizer and lr scheduler
    optimizer = torch.optim.AdamW(zachGPT.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    lr_scheduler = transformers.get_scheduler(
        "linear",
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=0,
    )
    if "scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Loading back data
    txt_set = TextDataset(data_file_path, model_config["max_seq_len"])
    dataload = DataLoader(
        txt_set, batch_size=64, shuffle=True, drop_last=True, pin_memory=True
    )

    running_loss = 0.0
    log_interval = 100

    print(f"Reprise de l'entraînement : pas {global_step} -> {total_steps}...")

    for x, y in dataload:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = zachGPT(x)
            loss = zachGPT.compute_loss(logits, y)

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
            ckpt = {
                "model_state_dict": zachGPT.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "global_step": global_step,
                "total_steps": total_steps,
                "loss": loss.item(),
                "model_config": model_config,
                "steps_history": steps_history,
                "loss_history": loss_history,
            }
            torch.save(ckpt, f"checkpoint_step_{global_step}.pt")

            if steps_history:
                plt.figure(figsize=(10, 5))
                plt.plot(
                    steps_history,
                    loss_history,
                    label="Loss (lissée)",
                    color="#2563eb",
                    linewidth=2,
                )
                plt.title(
                    f"zachGPT Pre-training Loss (Pas {global_step}/{total_steps})"
                )
                plt.xlabel("Pas d'optimisation")
                plt.ylabel("Perte Cross-Entropy")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.savefig("assets/loss_curve.png", dpi=150)
                plt.close()

        global_step += 1
        if global_step >= total_steps:
            break

    final_checkpoint = {
        "model_state_dict": zachGPT.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "global_step": global_step,
        "total_steps": total_steps,
        "loss": loss.item(),
        "model_config": model_config,
        "steps_history": steps_history,
        "loss_history": loss_history,
    }
    output_checkpoint_path = f"checkpoint_step_{global_step}.pt"
    torch.save(final_checkpoint, output_checkpoint_path)
    print(f"Entraînement terminé. Checkpoint enregistré : {output_checkpoint_path}")

    if steps_history:
        plt.figure(figsize=(10, 5))
        plt.plot(
            steps_history,
            loss_history,
            label="Loss finale",
            color="#16a34a",
            linewidth=2,
        )
        plt.title(
            f"zachGPT Pre-training Loss - Courbe finale ({global_step} pas)"
        )
        plt.xlabel("Pas d'optimisation")
        plt.ylabel("Perte Cross-Entropy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("assets/loss_curve.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    check_path = "checkpoint_step_20000.pt"
    data_path = "wikiset.bin"
    target = 25000  
    resume_training(check_path, data_path, target)