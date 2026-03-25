# 🧠 Fine-Tuning LLaMA 3.2 with Unsloth 

This project demonstrates how to **fine-tune a LLaMA 3.2 Instruct model using Unsloth**, apply **LoRA (PEFT)** for efficient training, and export the final model to **GGUF format** for fast local inference.

---

## 🚀 Overview

This pipeline covers:

* ⚡ Efficient fine-tuning using **Unsloth**
* 🧩 Parameter-efficient training using **LoRA (PEFT)**
* 📚 Training on reasoning-style dataset (**R1 Distill SFT**)
* 💬 Custom prompt formatting for reasoning tasks
* 🧠 Inference using Hugging Face `generate`
* 📦 Exporting model to **GGUF** for local deployment (llama.cpp, LM Studio, Ollama)

---

## 🏗️ Tech Stack

* Python 🐍
* PyTorch 🔥
* Transformers 🤗
* TRL (SFTTrainer)
* Unsloth ⚡
* Datasets 📊

---

## ⚙️ Model Details

* **Base Model**: `unsloth/Llama-3.2-3B-Instruct`
* **Quantization**: 4-bit (QLoRA)
* **Max Sequence Length**: 2048
* **Training Method**: Supervised Fine-Tuning (SFT)

---

## 🧪 Dataset

* Dataset: `ServiceNow-AI/R1-Distill-SFT`
* Fields used:

  * `problem`
  * `reannotated_assistant_content` (chain-of-thought)
  * `solution`

---

## 🧾 Prompt Format

```text
You are a reflective assistant engaging in thorough reasoning.

<problem>
{problem}
</problem>

{thought}
{solution}
```

---

## 🏋️ Training Configuration

| Parameter             | Value       |
| --------------------- | ----------- |
| Batch Size            | 2           |
| Gradient Accumulation | 4           |
| Learning Rate         | 2e-4        |
| Max Steps             | 60          |
| Optimizer             | AdamW 8-bit |
| LoRA Rank (r)         | 16          |

---

## 📦 Export to GGUF

Convert model for local inference:

```python
model.save_pretrained_gguf("rajat-001-3B-GGUF", tokenizer)
```

---

## 🧠 What is GGUF?

GGUF is a lightweight format for running LLMs locally using tools like:

* llama.cpp
* LM Studio
* Ollama

---

## ⚡ Run Locally (Example)

```bash
./main -m model.gguf -p "Explain AI in simple terms"
```

---

## 📬 Contact

For questions or collaboration, feel free to reach out!

---

⭐ If you found this helpful, consider starring the repo!
