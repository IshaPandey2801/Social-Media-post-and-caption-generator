# Social-Media-post-and-caption-generator
# Social Media Post & Caption Generator

A Generative AI–based web app that creates **social media captions and hashtags** from a short user prompt, with support for different **tones** and **platforms** (Instagram, LinkedIn, Twitter/X, etc.).  

The app uses:

- A base caption generator  
- A **fine-tuned T5 model** to rewrite captions according to tone & platform  
- A hashtag generator  
- A Gradio UI for easy interaction  
- A logging system (`output/result.csv`) to store generated examples

> **Note:** On a typical local laptop (without GPU), **image generation is disabled**. The focus is on captions + hashtags. Image generation can be run separately on GPU (e.g., Google Colab) if needed.

---

## 🔍 Features

- **Prompt-based caption generation**:  
  Enter a short idea like _"a child playing with toys on a rainy day"_ and generate a base caption.

- **Tone control**:  
  Choose from tones such as `playful`, `professional`, `motivational`, `romantic`, etc.

- **Platform-aware refinement**:  
  Select the target platform (`instagram`, `linkedin`, `twitter`, `facebook`, `youtube`).  
  A fine-tuned **T5 rewriter model** improves the caption based on tone and platform.

- **Hashtag generation**:  
  Generates simple hashtags from the refined caption.

- **Data logging for training**:  
  A **“Save example”** button stores prompt, base caption, refined caption, tone, platform and hashtags into `output/result.csv` for future analysis or fine-tuning.

---

## 🏗️ Project Architecture

High-level flow:

1. **User Input** (Gradio UI)  
   - Prompt / idea  
   - Tone  
   - Platform  

2. **Base caption generator**  
   - `backend/caption.py`  
   - Creates an initial caption from the prompt (tone/platform-aware if implemented inside).

3. **Fine-tuned caption rewriter**  
   - `backend/rewriter.py`  
   - Loads a fine-tuned **T5** model from `outputs/t5_rewriter/`  
   - Input: `base_caption + tone + platform`  
   - Output: refined caption

4. **Hashtag generator**  
   - `backend/hashtags.py`  
   - Generates simple hashtags based on the refined caption.

5. **(Optional) Image generator**  
   - `backend/generate.py`  
   - Currently returns `None` on local machine (image generation disabled due to hardware limits).  
   - Can be connected to a Stable Diffusion pipeline on GPU.

6. **Logging / Saving examples**  
   - `app.py` provides a "Save example" button  
   - Appends examples into `output/result.csv`.

---

## 📂 Project Structure

```text
Social-media-post-and-caption-generator/
├─ app.py                     # Main Gradio app
├─ backend/
│  ├─ generate.py             # Image generation (disabled locally)
│  ├─ caption.py              # Base caption generator
│  ├─ hashtags.py             # Hashtag generator
│  └─ rewriter.py             # Fine-tuned T5 caption rewriter
├─ output/
│  ├─ result.csv              # Logged examples (created by app)
│  ├─ train.jsonl             # Training data (input/output pairs)
│  ├─ train_augmented.jsonl   # Augmented (paraphrased) data
│  ├─ train_merged.jsonl      # Merged training data
│  └─ valid.jsonl             # Validation data
├─ outputs/
│  └─ t5_rewriter/            # Fine-tuned T5 model folder
│     ├─ config.json
│     ├─ pytorch_model.bin
│     ├─ tokenizer.json
│     ├─ tokenizer_config.json
│     └─ special_tokens_map.json
├─ fine_tune_rewriter.py      # Training script (used on Colab)
├─ requirements.txt
├─ README.md
└─ .gitignore
