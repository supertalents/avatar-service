from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
from PIL import Image
import torch
import os

os.environ['TRANSFORMERS_CACHE'] = './cache/'
os.environ['XDG_CACHE_HOME'] = './cache/'


controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe_canny = StableDiffusionControlNetPipeline.from_pretrained(
    "8glabs/realistic_vision_13", controlnet=controlnet_canny, safety_checker=None, torch_dtype=torch.float16
)
pipe_canny.scheduler = DDIMScheduler.from_config(pipe_canny.scheduler.config)

pipe_canny.enable_xformers_memory_efficient_attention()
device="cuda"
pipe_canny = pipe_canny.to(device)

print("successfully downloaded")
