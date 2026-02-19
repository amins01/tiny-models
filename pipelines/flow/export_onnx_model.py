import os
import torch
from accelerate import Accelerator
from ema_pytorch import EMA
import onnx

from flow.models.dit import TinyDiT

CHECKPOINT_DIR = "ckpts"
CHECKPOINT_NAME = "landscapes_256_x-v_iters_00125000"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_NAME}"
OUTPUT_PATH = f"{CHECKPOINT_NAME}_model.onnx"

# semantic
bs=1
model_dim=384
n_dit_layers=12
patch_size=16 # 8
image_channels=3
cond_channels=8 # 3
image_size=256 # 128
n_attn_heads=6
feed_fwd_dim=4*model_dim
txt_emb_dim=None
max_txt_len=None
n_adaln_cond_cls=None

dummy_x = torch.randn(bs, image_channels, image_size, image_size)
dummy_c = torch.randn(bs, cond_channels, image_size, image_size)
dummy_t = torch.rand(bs)
dummy_txt_c = None
dummy_txt_key_padding_mask = None
dummy_adaln_cond = None

# text edit
# bs=1
# model_dim=384
# n_dit_layers=12
# patch_size=8
# image_channels=3
# cond_channels=3
# image_size=128
# n_attn_heads=6
# feed_fwd_dim=4*model_dim
# txt_emb_dim=384
# max_txt_len=100
# n_adaln_cond_cls=None

# dummy_x = torch.randn(bs, image_channels, image_size, image_size)
# dummy_c = torch.randn(bs, cond_channels, image_size, image_size)
# dummy_t = torch.rand(bs)
# dummy_txt_c = torch.rand(bs, max_txt_len, txt_emb_dim)
# dummy_txt_key_padding_mask = torch.randint(0, 2, (bs, max_txt_len)).bool()
# dummy_adaln_cond = None

# art
# bs=1
# model_dim=384
# n_dit_layers=12
# patch_size=8
# image_channels=3
# cond_channels=3
# image_size=128
# n_attn_heads=6
# feed_fwd_dim=4*model_dim
# txt_emb_dim=None
# max_txt_len=None
# n_adaln_cond_cls=27

# dummy_x = torch.randn(bs, image_channels, image_size, image_size)
# dummy_c = torch.randn(bs, cond_channels, image_size, image_size)
# dummy_t = torch.rand(bs)
# dummy_adaln_cond = torch.tensor([0], dtype=torch.long)
# dummy_txt_c = None
# dummy_txt_key_padding_mask = None

model = TinyDiT(
    model_dim=model_dim,
    n_dit_layers=n_dit_layers,
    patch_size=patch_size,
    image_channels=image_channels,
    cond_channels=cond_channels,
    image_size=image_size,
    n_attn_heads=n_attn_heads,
    feed_fwd_dim=feed_fwd_dim,
    txt_emb_dim=txt_emb_dim,
    max_txt_len=max_txt_len,
    n_adaln_cond_cls=n_adaln_cond_cls
)

# load model
ema_model = EMA(model, beta=0.9999, update_every=10)
ema_path = os.path.join(CHECKPOINT_PATH, "custom_checkpoint_0.pkl")
print(f"Loading ema weights from {ema_path}...")

ema_state = torch.load(ema_path, map_location="cpu")
ema_model.load_state_dict(ema_state)

inference_model = ema_model.ema_model 
inference_model.eval()

temp_filename = "temp_export.onnx"

torch.onnx.export(
    inference_model,
    (dummy_x, dummy_t, dummy_c, dummy_txt_c, dummy_txt_key_padding_mask, dummy_adaln_cond),
    temp_filename,
    input_names=["x", "t", "cond"], # "txt_cond", "txt_key_padding_mask" # "adaln_cond"
    output_names=["velocity"],
    export_params=True,
)

model_proto = onnx.load(temp_filename)
onnx.save_model(model_proto, OUTPUT_PATH)

if os.path.exists(temp_filename):
    os.remove(temp_filename)
if os.path.exists(temp_filename + ".data"):
    os.remove(temp_filename + ".data")

print(f"Saved model to: {OUTPUT_PATH}")