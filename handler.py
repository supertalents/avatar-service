from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
from PIL import Image
import base64
from io import BytesIO
from app import cn
import torch
import runpod
import json


def handler(event):
    try:
        controlnet_canny = ControlNetModel.from_pretrained(
            "./cache", torch_dtype=torch.float16)
        pipe_canny = StableDiffusionControlNetPipeline.from_pretrained(
            "./cache", controlnet=controlnet_canny, safety_checker=None, torch_dtype=torch.float16
        )
        pipe_canny.scheduler = DDIMScheduler.from_config(
            pipe_canny.scheduler.config)

        pipe_canny.enable_xformers_memory_efficient_attention()
        device = "cuda"
        pipe_canny = pipe_canny.to(device)
        # print(event)
        data = event['input']
        prompt = data['prompt']
        no_of_images = int(data['no_of_images'])
        nprompt = data['nprompt']
        width = int(data['width'])
        height = int(data['height'])
        low_threshold = int(data['low_threshold'])
        high_threshold = int(data['high_threshold'])
        num_inference_steps = int(data['num_inference_steps'])
        guidance_scale = float(data['guidance_scale'])
        # ddim_steps = int(data['ddim_steps'])
        # bootstrapping = int(data['bootstrapping'])
        seed = int(data['seed'])
        image = data['image']
        # mask = data['mask']

        base64_bytes = base64.b64decode(image)
        image = Image.open(BytesIO(base64_bytes))

        image = cn(model=pipe_canny, prompt=prompt, image=image, height=height, width=width, num_inference_steps=num_inference_steps,
                   scale=guidance_scale, seed=seed, low_threshold=low_threshold, high_threshold=high_threshold,  nprompt=nprompt, no_of_images=no_of_images)

        buffered = BytesIO()
        # replace "JPEG" with the format of your image
        image.save(buffered, format="JPEG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # print(encoded_string)
        # print({'image': encoded_string})
        return {'image': encoded_string}

    except Exception as E:
        # print(E)
        return json.dumps({"message": "something went wrong" + str(E)}), 400


runpod.serverless.start({
    "handler": handler
})