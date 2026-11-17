"""Worst-case GPU memory check for training.

Builds one device batch from the 8 longest training prompts that pass the
shared length cap, pads each with a maximum-length completion (256 tokens),
and runs a forward + backward pass through the LoRA model. If this fits, no
batch in either arm can run out of memory. Also reports how many training
questions the cap drops.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from peft import get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src import data  # noqa: E402
from src.train_utils import CONFIGS, load_config, make_lora_config  # noqa: E402

base = load_config(CONFIGS / "lora.yaml")
tok = AutoTokenizer.from_pretrained(base["model"])
train = data.load_split("train")
lens = []
for e in train:
    ids = tok.apply_chat_template(data.build_messages(e), add_generation_prompt=True, tokenize=True)
    lens.append(len(ids if isinstance(ids, list) else ids["input_ids"]))
cap = base["max_prompt_tokens"]
kept = sorted(n for n in lens if n <= cap)
print(f"train questions: {len(lens)}; dropped by {cap}-token cap: {len(lens) - len(kept)}; "
      f"median {kept[len(kept) // 2]}, p99 {kept[int(len(kept) * .99)]}, max kept {kept[-1]}, max overall {max(lens)}")

B, L = 8, kept[-1] + 256
model = get_peft_model(AutoModelForCausalLM.from_pretrained(base["model"], dtype=torch.bfloat16),
                       make_lora_config()).cuda()
x = torch.randint(1000, 20000, (B, L), device="cuda")
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits
    loss = logits[:, -256:].float().log_softmax(-1).mean()
loss.backward()
print(f"worst-case batch {B} x {L} tokens: peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
      f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
print("MEM_TEST_OK")
