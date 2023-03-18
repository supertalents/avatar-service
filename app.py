import cv2
# import einops
# import gradio as gr
import numpy as np
import torch
from PIL import Image



# This command loads the individual model components on GPU on-demand. So, we don't
# need to explicitly call pipe.to("cuda").


# Generator seed,


# pose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
# controlnet_pose = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
# )
# pipe_pose = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, safety_checker=None, torch_dtype=torch.float16
# )
# pipe_pose.scheduler = UniPCMultistepScheduler.from_config(pipe_pose.scheduler.config)

# # This command loads the individual model components on GPU on-demand. So, we don't
# # need to explicitly call pipe.to("cuda").
# pipe_pose.enable_model_cpu_offload()

# # xformers
# pipe_pose.enable_xformers_memory_efficient_attention()


def get_canny_filter(image, low_threshold, high_threshold):
    
    if not isinstance(image, np.ndarray):
        image = np.array(image) 
        
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def cn(model,prompt, image, height, width , num_inference_steps, scale, seed, low_threshold, high_threshold,  nprompt):
    generator = torch.manual_seed(seed)


    # prompt = "photo of a person holding a cardboard in the hand, ((placeholder)) , ((cardboard in holding in hand))"
    canny_image = get_canny_filter(image, low_threshold, high_threshold)
    output = model(
        prompt=prompt,
        image=canny_image,
        negative_prompt=nprompt,
        height=height,
        width=width,
        guidance_scale=scale,
        generator=generator,
        num_images_per_prompt=1,
        num_inference_steps=num_inference_steps,
    )
    # return [canny_image,output.images[0]]
    return output.images[0]
    # print(image)
    # for i, img in enumerate(output.images[0]):
    #     img.save(f"out{i}.png")


    # Steps: 30, 
    # Sampler: DPM++ 
    # SDE Karras,
    #  CFG scale: 10, 
    #  Seed: 358924753, 
    #  Size: 512x512, 
    #  Model hash: ac32a2b5d7, Model: realisticvisionv1.3, 
    #  ControlNet Enabled: True, 
    #  ControlNet Module: canny, ControlNet Model: control_canny-fp16 [e3fe7712], 
    #  ControlNet Weight: 1, ControlNet Guidance Start: 0, ControlNet Guidance End: 1