import torch
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.tokenize import word_tokenize
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)
pipe.enable_attention_slicing()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_image(prompt):
    result = pipe(prompt, num_inference_steps=20, guidance_scale=7.5)
    return result.images[0]

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = caption_model.generate(**inputs, max_length=40)
    return processor.decode(output[0], skip_special_tokens=True)

def generate_hashtags(caption, max_tags=8):
    words = [w.lower() for w in word_tokenize(caption) if w.isalpha()]
    stop = {"the","and","a","in","on","of","is","to","for","with"}
    keywords = [w for w in words if w not in stop]
    top = [w for w,_ in Counter(keywords).most_common(max_tags)]
    return " ".join(["#" + w for w in top])
