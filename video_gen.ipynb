# --- سیل نمبر 1: ضروری لائبریریز انسٹال کرنے کے لیے ---
!pip install -q diffusers transformers accelerate torch

# --- سیل نمبر 2: ماڈل لوڈ کرنے اور ویڈیو بنانے کا کوڈ ---
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from google.colab import files

# ماڈل لوڈ کریں
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe.enable_model_cpu_offload()

# تصویر اپ لوڈ کریں
print("براہ کرم اپنی تصویر اپ لوڈ کریں جس کی ویڈیو بنانی ہے:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# تصویر کو تیار کریں
image = load_image(image_path)
image = image.resize((1024, 576))

# ویڈیو جنریٹ کریں
print("ویڈیو بن رہی ہے، اس میں 2 سے 5 منٹ لگ سکتے ہیں...")
generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

# ویڈیو ایکسپورٹ کریں
export_to_video(frames, "ai_video_output.mp4", fps=7)
files.download("ai_video_output.mp4")
print("ویڈیو ڈاؤن لوڈ ہو رہی ہے!")
