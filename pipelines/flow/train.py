import argparse
import os
import random
import shutil
import time
import numpy as np
import torch
import wandb
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from ema_pytorch import EMA

from flow.models.dit import TinyDiT
from dataset.utils import get_dataset
from sample import euler_sample

TXT_DATASETS = {"omni_edit", "gpt_edit"}
LOSS_SPACES = {"x", "v"}
PRED_SPACES = {"x", "v"}

def cycle(dl: torch.utils.data.DataLoader):
    while True:
        for data in dl:
            yield data

def sample(shape, device, sampling_type="uniform", mu=0, sigma=1):
    if sampling_type == "uniform":
        return torch.rand(shape, device=device)
    
    if sampling_type == "logit_normal":
        return torch.sigmoid(torch.randn(shape, device=device) * sigma + mu)
    
    raise ValueError(f"Invalid sampling type {sampling_type}")

def main(args):
    assert torch.cuda.is_available(), "No GPU detected"
    assert args.loss in LOSS_SPACES, f"Invalid loss {args.loss}"
    assert args.pred in PRED_SPACES, f"Invalid pred {args.pred}"

    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb")
    set_seed(args.seed + accelerator.process_index)

    exp_name = f"{args.dataset_name}__{time.time()}"
    exp_log_dir = os.path.join(args.log_dir, exp_name)
    if accelerator.is_main_process:
        os.makedirs(exp_log_dir, exist_ok=True)

    model = TinyDiT(
        model_dim=args.model_dim,
        n_dit_layers=args.n_dit_layers,
        patch_size=args.patch_size,
        image_channels=args.image_channels,
        cond_channels=args.cond_channels,
        image_size=args.image_size,
        n_attn_heads=args.n_attn_heads,
        feed_fwd_dim=args.feed_fwd_dim,
        txt_emb_dim=args.txt_emb_dim,
        max_txt_len=args.max_txt_len,
        n_adaln_cond_cls=args.n_adaln_cond_cls
    )
    model.train()

    if args.use_ema:
        ema_model = EMA(model, beta=args.ema_beta, update_every=10)
        ema_model.to(accelerator.device)
        accelerator.register_for_checkpointing(ema_model)

    n_params = sum([p.numel() for p in model.parameters()])
    print(f"DiT model has {n_params:,} params")

    # TODO: set up resume training from ckpt

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.adam_b1, args.adam_b2),
        weight_decay=0.05,
    )

    palette = None
    if args.dataset_name == "landscapes":
        palette = np.array([
            [135, 206, 235], # sky
            [34, 139, 34], # vegetation
            [112, 128, 144], # rock
            [0, 0, 255], # water
            [70, 70, 70], # road
            [238, 214, 175], # sand
        ], dtype=np.uint8)
    elif args.dataset_name in TXT_DATASETS:
        from transformers import AutoTokenizer, AutoModel
        txt_tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
        txt_tokenizer = AutoTokenizer.from_pretrained(txt_tokenizer_name)
        txt_encoder = AutoModel.from_pretrained(txt_tokenizer_name)
        txt_encoder.to(accelerator.device)
        txt_encoder.eval()

    per_gpu_bs = int(args.bs // dist.get_world_size())
    dataset = get_dataset(args.dataset_name, args)

    val_size = 250
    rest_size = len(dataset) - val_size
    val_dataset, train_dataset = random_split(dataset, [val_size, rest_size])

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    train_data_loader = cycle(train_data_loader)

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=min(per_gpu_bs, args.visualize_num),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    val_data_loader = cycle(val_data_loader)
    val_data_loader = iter(val_data_loader)

    print(f"Train dataset contains {len(train_dataset)} samples.")
    print(f"Val dataset contains {len(val_dataset)} samples.")

    model, optimizer, train_data_loader = accelerator.prepare(model, optimizer, train_data_loader)

    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    if accelerator.is_main_process:
        print("WANDB")
        accelerator.init_trackers(
            project_name=args.wandb_project_name,
            init_kwargs={
                "wandb": {
                    "entity": args.wandb_entity,
                    "config": vars(args),
                    "name": exp_name,
                    "dir": exp_log_dir,
                }
            },
        )

    running_loss, running_grad_norm, start_time = 0, 0, time.time()
    cur_step = 0
    while cur_step < args.train_steps:
        with accelerator.autocast():
            model.train()

            txt_cond = None
            txt_key_padding_mask = None
            adaln_cond = None
            
            batch_data = next(train_data_loader)
            if args.dataset_name in TXT_DATASETS:
                cond, img, txt = batch_data
                txt_tokens = txt_tokenizer(txt, padding="max_length", truncation=True, max_length=args.max_txt_len, return_tensors="pt")
                txt_tokens = txt_tokens.to(accelerator.device)
                txt_key_padding_mask = txt_tokens["attention_mask"] == 0
                with torch.no_grad():
                    txt_cond = txt_encoder(**txt_tokens)[0]
            elif args.dataset_name == "wikiart":
                cond, img, adaln_cond = batch_data
                adaln_cond = adaln_cond.to(accelerator.device)
            else:
                cond, img = batch_data
            
            cond, img = cond.to(accelerator.device), img.to(accelerator.device)

            t = sample((per_gpu_bs,), accelerator.device, args.t_sampling, args.t_logit_normal_mu, args.t_logit_normal_sigma)
            eps = torch.randn_like(img)
            expanded_t = t.repeat(-1, 1, 1, 1)
            noisy_img = expanded_t * img + (torch.ones_like(expanded_t) - expanded_t) * eps
            model_out = model(noisy_img, t, cond, txt_cond, txt_key_padding_mask, adaln_cond)

            # prepare target
            if args.loss == "x":
                target = img
            elif args.loss == "v":
                target = img - eps # (img - noisy_img) / ((torch.ones_like(expanded_t) - expanded_t)).clip(0.05)
            
            # prepare pred
            if args.pred == args.loss:
                pred = model_out
            elif args.pred == "x" and args.loss == "v":
                pred = (model_out - noisy_img) / ((torch.ones_like(expanded_t) - expanded_t)).clip(0.05) # vθ=(xθ−zt)/(1−t)
            elif args.pred == "v" and args.loss == "x":
                pred = (torch.ones_like(expanded_t) - expanded_t) * model_out + noisy_img # xθ=(1−t)vθ+zt
            
            loss = torch.nn.functional.mse_loss(pred, target)
            # print(f"{cur_step}. Loss", loss)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm) # float("inf")

            # grad_norm = torch.tensor(0.0, device=accelerator.device)

            optimizer.step()
            if args.use_ema:
                ema_model.update()
            optimizer.zero_grad()

            running_loss += accelerator.gather(loss).mean().item()
            running_grad_norm += accelerator.gather(grad_norm).mean().item()

            cur_step += 1

            if accelerator.is_main_process and accelerator.sync_gradients:
                sampling_model = ema_model.ema_model if args.use_ema else model
                sampling_model.eval()

                if cur_step % args.log_every == 0:
                    average_loss = torch.tensor(running_loss / args.log_every, device=accelerator.device).item()
                    average_grad_norm = torch.tensor(running_grad_norm / args.log_every, device=accelerator.device).item()

                    end_time = time.time()
                    average_time = (end_time - start_time) / args.log_every
                    start_time = time.time()

                    print(f"Step {cur_step:08d} | Loss {average_loss:.4f} | Time {average_time:.4f}s | Grad Norm {average_grad_norm:.4f}")
                    
                    running_loss = 0
                    running_grad_norm = 0

                    log_dict = {
                        "loss": average_loss,
                        "grad_norm": average_grad_norm,
                    }

                    accelerator.log(log_dict, step=cur_step)

                if (args.visualize_first_step and cur_step == 1) or (cur_step % args.visualize_every == 0):
                    print("Visualizing...")
                    with torch.no_grad():
                        val_generated_count = 0
                        samples_to_log = {}
                        while val_generated_count < args.visualize_num:
                            val_txt_emb = None
                            val_txt_key_padding_mask = None
                            all_val_txt = None
                            all_val_adaln_conds = None
                            batch_data = next(val_data_loader)

                            if args.dataset_name in TXT_DATASETS:
                                all_val_conds, all_val_imgs, all_val_txt = batch_data
                                val_txt_tokens = txt_tokenizer(all_val_txt, padding="max_length", truncation=True, max_length=args.max_txt_len, return_tensors="pt")
                                val_txt_tokens = val_txt_tokens.to(accelerator.device)
                                val_txt_key_padding_mask = val_txt_tokens["attention_mask"] == 0
                                val_txt_emb = txt_encoder(**val_txt_tokens)[0]
                            elif args.dataset_name == "wikiart":
                                all_val_conds, all_val_imgs, all_val_adaln_conds = batch_data
                                all_val_adaln_conds = all_val_adaln_conds.to(accelerator.device)
                            else:
                                all_val_conds, all_val_imgs = batch_data
                            
                            all_val_conds, all_val_imgs = all_val_conds.to(accelerator.device), all_val_imgs.to(accelerator.device)
                            all_val_img_preds = euler_sample(
                                sampling_model,
                                all_val_conds,
                                val_txt_emb,
                                val_txt_key_padding_mask,
                                all_val_adaln_conds,
                                args.max_sampling_t,
                                args.pred,
                            )
                            val_generated_count += all_val_img_preds.shape[0]

                            for i in range(all_val_img_preds.shape[0]):
                                val_img = all_val_imgs[i].permute(1, 2, 0).cpu().numpy()
                                val_img = np.clip((val_img + 1.0) / 2.0, 0.0, 1.0)
                                val_img = (val_img * 255).astype(np.uint8)

                                val_img_pred = all_val_img_preds[i].permute(1, 2, 0).cpu().numpy()
                                val_img_pred = np.clip((val_img_pred + 1.0) / 2.0, 0.0, 1.0)
                                val_img_pred = (val_img_pred * 255).astype(np.uint8)

                                if args.dataset_name == "landscapes":
                                    mask_indices = torch.argmax(all_val_conds[i], dim=0).cpu().numpy().astype(np.uint8)
                                    val_cond = palette[mask_indices]
                                else:
                                    val_cond = all_val_conds[i].permute(1, 2, 0).cpu().numpy()
                                    val_cond = np.clip((val_cond + 1.0) / 2.0, 0.0, 1.0)
                                    val_cond = (val_cond * 255).astype(np.uint8)

                                val_img = Image.fromarray(val_img)
                                val_img_pred = Image.fromarray(val_img_pred)
                                val_cond = Image.fromarray(val_cond)
                            
                                img_display = val_img.resize((256, 256), Image.NEAREST)
                                pred_img_display = val_img_pred.resize((256, 256), Image.NEAREST)
                                cond_pil_display = val_cond.resize((256, 256), Image.NEAREST)

                                cond_caption = "cond"
                                
                                if all_val_txt is not None:
                                    cond_caption = f"prompt: {all_val_txt[i]}"
                                
                                if all_val_adaln_conds is not None:
                                    style = val_dataset.dataset.style_cls_to_txt[all_val_adaln_conds[i].item()]
                                    cond_caption += f"\ncls: {style}"

                                samples_to_log[f"gen_{val_generated_count + i}"] = [
                                    wandb.Image(cond_pil_display, caption=cond_caption),
                                    wandb.Image(pred_img_display, caption="generated"),
                                    wandb.Image(img_display, caption="original")
                                ]
                        
                        accelerator.log({**samples_to_log}, step=cur_step)
                
                if cur_step % args.ckpt_every == 0 and accelerator.is_main_process:
                    ckpt_path = os.path.join(args.checkpoint_dir, exp_name, f"iters_{cur_step:08d}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    accelerator.save_state(ckpt_path)
                    print(f"Saved Iter {cur_step} checkpoint to {ckpt_path}")

                    for ckpt_dir in os.listdir(os.path.join(args.checkpoint_dir, exp_name)):
                        if ckpt_dir.startswith("iters") and ckpt_dir != f"iters_{cur_step:08d}":
                            save_iter = int(ckpt_dir.split("_")[-1])
                            if save_iter < cur_step - args.keep_last_k * args.ckpt_every:
                                if save_iter not in [5e4, 1e5, 2e5, 3e5]:
                                    shutil.rmtree(os.path.join(args.checkpoint_dir, exp_name, ckpt_dir))

            accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        final_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name, f"iters_{cur_step:08d}_final")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        accelerator.save_state(final_ckpt_dir)
        print(f"Saved Final Iter {cur_step} checkpoint to {final_ckpt_dir}")
    
    accelerator.wait_for_everyone()
    print("Training Done.")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--train_steps", type=int, default=300000)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_b1", type=float, default=0.9)
    parser.add_argument("--adam_b2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataset_name", type=str, default="landscapes")
    parser.add_argument("--dataset_path", type=str, default="data/landscapes")
    parser.add_argument("--mask_path", type=str, default="")
    parser.add_argument("--metadata_path", type=str, default="")
    parser.add_argument("--pred", type=str, choices=["x", "v"], default="v")
    parser.add_argument("--loss", type=str, choices=["x", "v"], default="v")
    parser.add_argument("--t_sampling", type=str, choices=["uniform", "logit_normal"], default="uniform")
    parser.add_argument("--t_logit_normal_mu", type=float, default=-0.8)
    parser.add_argument("--t_logit_normal_sigma", type=float, default=1)
    
    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_beta", type=float, default=0.9999)
    
    # architecture
    parser.add_argument("--model_dim", type=int, default=384)
    parser.add_argument("--n_dit_layers", type=int, default=12)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--cond_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--feed_fwd_dim", type=int, default=4*384)
    parser.add_argument("--txt_emb_dim", type=int, default=None)
    parser.add_argument("--max_txt_len", type=int, default=None)
    parser.add_argument("--n_adaln_cond_cls", type=int, default=None)

    # sampling
    parser.add_argument("--max_sampling_t", type=int, default=10)

    # logging
    parser.add_argument("--log_dir", type=str, default="out")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--visualize_first_step", action="store_true", default=True)
    parser.add_argument("--visualize_every", type=int, default=250)
    parser.add_argument("--visualize_num", type=int, default=1)
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project_name", type=str, default="TinyDiT")

    # checkpoints
    parser.add_argument("--ckpt_every", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="ckpts")
    parser.add_argument("--keep_last_k", type=int, default=10)

    args = parser.parse_args()

    main(args)