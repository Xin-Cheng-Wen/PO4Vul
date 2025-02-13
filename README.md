# PO4Vul

### In training and inference phase:
We provide config about fine-tuning and IPO for LLMs in `src\LLM`.

Make sure to execute these commands in the `LLaMA-Factory` (V0.9.2) directory.

#### SFT

```bash
llamafactory-cli train Src/LLM/SFT/qwen2.5_full_sft.yaml
```

#### IPO

```bash
llamafactory-cli train Src/LLM/IPO/qwen2.5_lora_ipo_C0.yaml
llamafactory-cli train Src/LLM/IPO/qwen2.5_lora_ipo_C.yaml
```

### In BVD phase:
We provide src for Qwen2.5-Coder-32B in `src\GenData\sft_gen.py`.

### In COPO phase:
We provide src for Qwen2.5-Coder-32B in `src\GenData\ins_c_step_{C}.py` and `src\GenData\task{C}.py`. "C" denotes the step of generation.

### Eval phase:
We provide src for Vulnerability Detection in `Eval`.
