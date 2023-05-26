import numpy as np
import torch

class DDPMScheduler():
    """
    Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
    Langevin dynamics sampling.

    For more details, see the original paper: https://arxiv.org/abs/2006.11239
    """
    
    def __init__(self,beta_start=1e-4, beta_end=1e-2, num_train_timesteps=1000):
        # Betas settings are in section 4 of https://arxiv.org/pdf/2006.11239.pdf
        # Implemented linear schedule for now, cosine works better tho.
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1 - self.betas

        # alpha-hat in the paper, precompute them
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

        self.num_train_timesteps = num_train_timesteps

    def set_timesteps(self, num_inference_steps: int):
        #1000：num_train_timesteps
        self.timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=int)
    
    def step(self, model_output, i, sample):
        timestep = self.timesteps[i]
        timestep_prev = self.timesteps[i - 1] if i > 0 else self.timesteps[0]

        beta_t = self.betas[timestep]
        alpha_t = self.alphas[timestep]
        alpha_hat_t = self.alphas_hat[timestep]
        alpha_hat_prev = self.alphas_hat[timestep_prev]

        # Algorithm 2, step 4: calculate x_{t-1} with alphas and variance.
        # Since paper says we can use fixed variance (section 3.2, in the beginning),
        # we will calculate the one which assumes we have x0 deterministically set to one point.
        beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat_t) * beta_t
        variance = torch.sqrt(beta_t_hat) * torch.randn(sample.shape) if i > 0 else torch.zeros(sample.shape)

        sample_prev = torch.pow(alpha_t, -0.5) * (sample -
                                            beta_t / torch.sqrt((1 - alpha_hat_t)) *
                                            model_output) + variance.to(model_output.device)
        
        return sample_prev
