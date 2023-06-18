import torch
from ldm.models.autoencoder import AutoencoderKL
from image_helper import load_image, save_image
from safetensors.torch import load_file

#VAE模型
def load_vae():
    #初始化模型
    init_config = {
        "embed_dim": 4,
        "monitor": "val/rec_loss",
        "ddconfig":{
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult":[1,2,4,4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0,
        },
        "lossconfig":{
          "target": "torch.nn.Identity"
        }
    }
    vae = AutoencoderKL(**init_config)
    #加载预训练参数
    #pl_sd = torch.load("../models/sd-v1-4.ckpt", map_location="cpu")
    #sd = pl_sd["state_dict"]
    sd = load_file("F:/repo/stable-diffusion-stepbystep/models/deliberate_v2.safetensors")
    model_dict = vae.state_dict()
    for k, v in model_dict.items():
        model_dict[k] = sd["first_stage_model."+k]
    vae.load_state_dict(model_dict, strict=False)

    vae.eval()
    return vae

#测试vae模型
def test_vae():
    vae = load_vae()
    img = load_image("demo.png")  #(1,3,512,512)   
    latent = vae.encode(img).sample()       #(1,4,64,64)
    samples = vae.decode(latent)            #(1,3,512,512)
    save_image(samples,"vae.png")

#test_vae()