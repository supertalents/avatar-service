from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import time
import base64
from io import BytesIO
from app import cn
import random
import threading
from PIL import Image
import torch
import os

os.environ['TRANSFORMERS_CACHE'] = './cache/'
os.environ['XDG_CACHE_HOME'] = './cache/'


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
            "8glabs/realistic_vision_13", controlnet=controlnet_canny, safety_checker=None, torch_dtype=torch.float16
        )
        pipe_canny.scheduler = DDIMScheduler.from_config(pipe_canny.scheduler.config)

        pipe_canny.enable_xformers_memory_efficient_attention()
        device="cuda"
        pipe_canny = pipe_canny.to(device)

        
@app.route("/", methods=['GET'])
def index():
    return "Live"

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
        # mask = data['mask']

        base64_bytes = base64.b64decode(image)
        image = Image.open(BytesIO(base64_bytes))

        image = cn(model=pipe_canny,prompt=prompt, image=image ,height=height, width=width , num_inference_steps=num_inference_steps, scale=guidance_scale, seed=seed, low_threshold=low_threshold, high_threshold=high_threshold,  nprompt=nprompt)


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
