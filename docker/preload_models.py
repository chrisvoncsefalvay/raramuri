"""Download all HF models into the image cache at build time."""
import os
import sys

os.environ.setdefault("HF_HOME", "/models/hf")
os.environ.setdefault("TRIBE_CACHE", "/models/tribe")
os.environ.setdefault("MNE_DATA", "/models/mne_data")

from huggingface_hub import snapshot_download
from transformers import AutoModel

token = os.environ.get("HF_TOKEN")
if not token:
    print("FATAL: HF_TOKEN required", file=sys.stderr)
    sys.exit(1)

hf_home = os.environ["HF_HOME"]

for name in ("facebook/w2v-bert-2.0", "meta-llama/Llama-3.2-3B"):
    print(f"Downloading {name}...")
    AutoModel.from_pretrained(name, cache_dir=hf_home, token=token)

vjepa = "facebook/vjepa2-vitg-fpc64-256"
print(f"Downloading {vjepa}...")
try:
    AutoModel.from_pretrained(vjepa, cache_dir=hf_home, token=token)
except Exception:
    snapshot_download(vjepa, cache_dir=hf_home, token=token)

# TRIBEv2 weights
print("Downloading TRIBEv2...")
from tribev2 import TribeModel
TribeModel.from_pretrained("facebook/tribev2", cache_folder=os.environ["TRIBE_CACHE"])

# Parakeet
print("Downloading Parakeet...")
import nemo.collections.asr as nemo_asr
nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

print("All models downloaded.")
