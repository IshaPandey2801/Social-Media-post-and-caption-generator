import csv
from pathlib import Path
from datetime import datetime

import gradio as gr

# ===== Backend imports with safe fallbacks =====

# Image generator (currently disabled locally – returns None in backend/generate.py)
try:
    from backend.generate import generate_image
except Exception:
    def generate_image(prompt, steps=20, guidance=7.5):
        return None

# Caption generator
try:
    from backend.caption import generate_caption
except Exception:
    def generate_caption(prompt: str, tone: str = "", platform: str = "") -> str:
        # Fallback simple caption if backend.caption is missing
        extra = []
        if tone:
            extra.append(f"tone: {tone}")
        if platform:
            extra.append(f"platform: {platform}")
        extra_str = " (" + ", ".join(extra) + ")" if extra else ""
        return f"Generated caption for: {prompt}{extra_str}"

# Hashtag generator
try:
    from backend.hashtags import generate_hashtags
except Exception:
    def generate_hashtags(text: str, tone: str = "", platform: str = "") -> str:
        # Fallback simple hashtags
        base = text.lower().replace(".", "").replace(",", "")
        words = base.split()[:4]
        tags = [f"#{w}" for w in words]
        return " ".join(tags) if tags else "#socialmedia #post"

# Rewriter (fine-tuned T5)
try:
    from backend.rewriter import rewrite_caption
except Exception:
    def rewrite_caption(base_caption: str, tone: str = "", platform: str = "") -> str:
        # Fallback: just echo base caption
        extra = []
        if tone:
            extra.append(f"tone: {tone}")
        if platform:
            extra.append(f"platform: {platform}")
        extra_str = " (" + ", ".join(extra) + ")" if extra else ""
        return base_caption + extra_str


# ===== Logging / CSV setup =====

RESULTS_CSV = Path("output") / "result.csv"


def save_example_to_csv(
    prompt: str,
    tone: str,
    platform: str,
    base_caption: str,
    final_caption: str,
    hashtags: str,
):
    """
    Append one example to output/result.csv and return a status message.
    Image path is left empty because local image generation is disabled.
    """
    RESULTS_CSV.parent.mkdir(exist_ok=True)

    # Ensure header exists
    if not RESULTS_CSV.exists():
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "image_path",
                    "prompt_text",
                    "base_caption",
                    "final_caption",
                    "tone",
                    "platform",
                    "hashtags",
                    "timestamp",
                ]
            )

    # Find last id (from last row)
    last_id = 0
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        if len(rows) >= 2:
            try:
                last_id = int(rows[-1][0])
            except Exception:
                last_id = 0

    new_id = last_id + 1
    timestamp = datetime.now().isoformat(timespec="seconds")
    image_path = ""  # no image stored locally

    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                new_id,
                image_path,
                prompt,
                base_caption,
                final_caption,
                tone,
                platform,
                hashtags,
                timestamp,
            ]
        )

    return f"Saved example #{new_id} at {timestamp}"


# ===== Core pipeline used by Gradio =====

def pipeline(prompt: str, tone: str, platform: str, steps: int, guidance: float):
    """
    Main function called by the 'Generate' button.
    1. Generate base caption from prompt.
    2. Rewrite caption using fine-tuned model.
    3. Generate hashtags.
    4. (Optionally) generate image – currently disabled, will return None.
    """
    if not prompt or prompt.strip() == "":
        return None, "Please enter a prompt.", "", ""

    # 1) Base caption
    base_caption = generate_caption(prompt, tone=tone, platform=platform)

    # 2) Rewritten caption (using fine-tuned T5)
    final_caption = rewrite_caption(base_caption, tone=tone, platform=platform)

    # 3) Hashtags (based on final caption + tone/platform)
    hashtags = generate_hashtags(final_caption, tone=tone, platform=platform)

    # 4) Image (currently disabled locally; function returns None)
    img = generate_image(prompt, steps=steps, guidance=guidance)

    return img, base_caption, final_caption, hashtags


# ===== Gradio UI =====

with gr.Blocks(title="Social Media Post & Caption Generator") as demo:
    gr.Markdown(
        """
        # Social Media Post & Caption Generator
        
        - Enter a **prompt / idea** for your post.
        - Choose a **tone** and **platform**.
        - Click **Generate** to get:
          - (Optional) Image *(disabled on this device)*
          - Base caption
          - Refined caption (fine-tuned model)
          - Hashtags
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="Post Idea / Prompt",
                placeholder="e.g. A child playing with toys on a rainy day",
                lines=3,
            )
            tone_dropdown = gr.Dropdown(
                label="Tone",
                choices=["playful", "professional", "motivational", "romantic", "informative"],
                value="playful",
            )
            platform_dropdown = gr.Dropdown(
                label="Platform",
                choices=["instagram", "linkedin", "twitter", "facebook", "youtube"],
                value="instagram",
            )
            steps_slider = gr.Slider(
                label="Image steps (ignored locally)",
                minimum=10,
                maximum=40,
                value=20,
                step=1,
            )
            guidance_slider = gr.Slider(
                label="Image guidance scale (ignored locally)",
                minimum=3.0,
                maximum=15.0,
                value=7.5,
                step=0.5,
            )

            generate_btn = gr.Button("Generate", variant="primary")

            save_btn = gr.Button("Save example", variant="secondary")
            save_status = gr.Textbox(
                label="Save status",
                interactive=False,
                lines=1,
                placeholder="Click 'Save example' after generation...",
            )

        with gr.Column(scale=1):
            image_out = gr.Image(
                label="Generated Image (disabled locally – may remain blank)",
                height=256,
            )
            base_caption_box = gr.Textbox(
                label="Base Caption",
                lines=3,
            )
            final_caption_box = gr.Textbox(
                label="Refined Caption (Fine-tuned model)",
                lines=3,
            )
            hashtags_box = gr.Textbox(
                label="Hashtags",
                lines=2,
            )

    # Wire buttons to functions
    generate_btn.click(
        fn=pipeline,
        inputs=[prompt_box, tone_dropdown, platform_dropdown, steps_slider, guidance_slider],
        outputs=[image_out, base_caption_box, final_caption_box, hashtags_box],
    )

    save_btn.click(
        fn=save_example_to_csv,
        inputs=[
            prompt_box,         # prompt
            tone_dropdown,      # tone
            platform_dropdown,  # platform
            base_caption_box,   # base caption
            final_caption_box,  # final caption
            hashtags_box,       # hashtags
        ],
        outputs=save_status,
    )


if __name__ == "__main__":
    
    demo.launch(share=True)
