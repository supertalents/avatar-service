import cv2
import numpy as np
import torch
from PIL import Image


def get_canny_filter(image, low_threshold, high_threshold):
    
    if not isinstance(image, np.ndarray):
        image = np.array(image) 
        
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def cn(model,prompt, image, height, width , num_inference_steps, scale, seed, low_threshold, high_threshold,  nprompt, no_of_images=1):
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
        num_images_per_prompt=no_of_images,
        num_inference_steps=num_inference_steps,
        # mask_image=mask
    )
    # return [canny_image,output.images[0]]
    return output.images
