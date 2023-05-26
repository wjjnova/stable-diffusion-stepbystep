import torch
from clip_helper import prompts_embedding
from image_helper import load_image, save_image
from scheduler_lms import LMSScheduler
from unet_helper import load_unet
from vae_helper import load_vae

def img2img():
    #unet
    unet = load_unet().cuda()
    #调度器
    scheduler = LMSScheduler()
    scheduler.set_timesteps(100)
    #文本编码
    prompts = ["a dog and a cat"]
    text_embeddings = prompts_embedding(prompts)
    text_embeddings = text_embeddings.cuda()     #(1, 77, 768)
    uncond_prompts = [""]
    uncond_embeddings = prompts_embedding(uncond_prompts)
    uncond_embeddings = uncond_embeddings.cuda() #(1, 77, 768)
    #VAE
    vae = load_vae()
    init_img = load_image("txt2img.png")
    init_latent = vae.encode(init_img).sample().cuda()*0.18215
    #初始隐变量
    noise_latents = torch.randn( (1, 4, 64, 64),device="cuda")
    START_STRENGTH = 45
    print("xxxx init_latent ",init_latent.shape)
    print("xxxx noise_latents ",noise_latents.shape)
    latents = init_latent + noise_latents*scheduler.sigmas[START_STRENGTH]
    #循环步骤

    for i, t in enumerate(scheduler.timesteps):  #[999.  988.90909091 978.81818182 ...100个
        print(i,t)
        if i < START_STRENGTH:
            continue
        latent_model_input = latents  #torch.Size([1, 4, 64, 64])  
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
        
        timestamp = torch.tensor([t])

        with torch.no_grad(): 
            noise_pred_text = unet(latent_model_input.cuda(), timestamp.cuda(), text_embeddings.cuda())
            noise_pred_uncond = unet(latent_model_input.cuda(), timestamp.cuda(), uncond_embeddings.cuda())
            guidance_scale = 7.5 
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, i, latents)

    latents = 1 / 0.18215 * latents
    image = vae.decode(latents.cpu())
    save_image(image,"img2img.png")

img2img()
