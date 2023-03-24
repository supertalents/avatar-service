## REST API FOR CONTROL-NET

This code uses realisticv1.3 model as a base..

Thanks to 

```
#create env
Python -m venv venv
#activate env
source venv/bin/activate
pip install -r requirements.txt
#now run the main.py file
python main.py
```


You will do post request to IP
```
IP : 0.0.0.0:5000/controlnet
```
you will post json with it
```
{
	"prompt" : "(RAW photo), (superhero), ((((laser eyes)))), (black eye mask), good looking, (blue, purple, red neon background colors:0.8), (high detailed skin:1.2), (8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3)",
	"nprompt" : "(((nsfw))), (((no shirt))), ((((dark, evil, devil, demon)))), ((weird eyes)), (((((backlighting, halation, lens flare, glare))))), ((((ugly, bad face, evil face)))), (((((face mask, big mask, head mask, mouth mask, mask on head, weird mask, helmet))))), ((covered nose, covered forehead)), ((face paint)), ((((light on face, glowing face, glowing mouth, light on mouth)))), windows, canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), (((duplicate))), ((((morbid)))), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, ((((cross-eye)))), body out of frame, blurry, bad art, bad anatomy, 3d render",
	"width" : "512",
	"height" : "512",
	"num_inference_steps" : "35",
	"low_threshold" : "100",
	"high_threshold" : "200",
	"guidance_scale" : "10",
	"seed" : "358924753",
	"image" : "base64StringImage"
}
```
And in return you will get a response in json
```
	{ "image": "base64StringImage" }
	
	

## Update 25 MARCH, 2023

Docker Image available for deploying on runpod.io

```
docker pull pydashninja/multi-region
```

Command to run in runpod.io:


```
docker run --gpus all -p 5000:443 multi-region
```

Enjoy!
