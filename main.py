from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
# from flask_cors import CORS
import time
import base64
from io import BytesIO
from app import cn
import random
import threading
from PIL import Image
import torch


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# lock = threading.Lock()

@app.before_first_request
def loading():
    # Models
    global pipe_canny
    try:
        if next(pipe_canny.parameters()).is_cuda:
            pass
        # return jsonify({"message" : "Success"}), 200   
    except:
        controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        pipe_canny = StableDiffusionControlNetPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V1.3_Fantasy.ai", controlnet=controlnet_canny, safety_checker=None, torch_dtype=torch.float16
        )
        # pipe_canny.scheduler = UniPCMultistepScheduler.from_config(pipe_canny.scheduler.config)
        pipe_canny.scheduler = DDIMScheduler.from_config(pipe_canny.scheduler.config)
        pipe_canny.enable_model_cpu_offload()

        pipe_canny.enable_xformers_memory_efficient_attention()
        device="cuda"
        pipe_canny = pipe_canny.to(device)


@app.route('/controlnet',methods=['POST'])
@cross_origin()
def get_image():
  # with lock:
    try:
    #   # print("hello")
        data = request.get_json()
    #   if data['operation'] == 'sd':
        prompt = data['prompt']
        nprompt = data['nprompt']
        width= int(data['width']) 
        height= int(data['height'])
        low_threshold = int(data['low_threshold'])
        high_threshold = int(data['high_threshold']) 
        num_inference_steps = int(data['num_inference_steps'])
        guidance_scale= float(data['guidance_scale'])
        # ddim_steps = int(data['ddim_steps'])
        # bootstrapping = int(data['bootstrapping'])
        seed = int(data['seed'])
        image = data['image']

        base64_bytes = base64.b64decode(image)
        image = Image.open(BytesIO(base64_bytes))

        image = cn(model=pipe_canny,prompt=prompt, image=image, height=height, width=width , num_inference_steps=num_inference_steps, scale=guidance_scale, seed=seed, low_threshold=low_threshold, high_threshold=high_threshold,  nprompt=nprompt)
        # cn(model, sampler ,prompt, image, image_resolution, ddim_steps, scale, seed, eta, low_threshold, high_threshold,  nprompt):
        # image = multsd(sd, mask=image)    # sd = MultiDiffusion(device, opt.sd_version)


        buffered = BytesIO()
        image.save(buffered, format="JPEG") # replace "JPEG" with the format of your image
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # print(encoded_string)
        # print({'image': encoded_string})
        return {'image': encoded_string}

    except Exception as E:
        print(E)
        return jsonify({"message" : "soemthing went wrong" + str(E)}), 400


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)