import torch
from clip_helper import prompts_embedding
from image_helper import save_image
from scheduler_ddim import DDIMScheduler
from unet_helper import load_unet
from vae_helper import load_vae

def txt2img_ddim():
    #unet
    unet = load_unet()
    #调度器
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(20)
    #文本编码
    prompts = ["Jane Eyre with headphones, natural skin texture, 24mm, 4k textures, soft cinematic light, adobe lightroom, photolab, hdr, intricate, elegant, highly detailed, sharp focus, ((((cinematic look)))), soothing tones, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, dim colors, exposure blend, hdr, faded"]
    #prompts = ["cherry blossom, ultra detailed, cinematic lighting, HDR, ilustration, landsape, sunrise, impressive, chill, inspirational"]
    text_embeddings = prompts_embedding(prompts)
    text_embeddings = text_embeddings.cuda()     #(1, 77, 768)
    uncond_prompts = ["(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"]
    #uncond_prompts = [""]
    uncond_embeddings = prompts_embedding(uncond_prompts)
    uncond_embeddings = uncond_embeddings.cuda() #(1, 77, 768)
    #初始隐变量
    latents = torch.randn((1, 4, 64, 64))  #(1, 4, 64, 64)
    #latents = torch.zeros((1, 4, 64, 64))  #(1, 4, 64, 64)
    latents = latents.cuda()
    #循环步骤
    for i, t in enumerate(scheduler.timesteps):  #timesteps=[999.  988.90909091 978.81818182 ...100个
        timestamp = torch.tensor([t]).cuda()

        with torch.no_grad():
            noise_pred_text = unet(latents, timestamp, text_embeddings)
            noise_pred_uncond = unet(latents, timestamp, uncond_embeddings)
            guidance_scale = 7.5
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents)    
    latents = 1 / 0.18215 * latents

    vae = load_vae()
    image = vae.decode(latents.cpu())  #(1, 3, 512, 512)
    #image = (image / 2 + 0.5).clamp(0, 1)
    save_image(image,"txt2img_ddim.png")

txt2img_ddim()
