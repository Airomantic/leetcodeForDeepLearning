import numpy as np

import matplotlib.pyplot as plt


class SimpleDiffusionModel:
    # beta_start: Noise intensity at the beginning of the diffusion process
    def __init__(self, timesteps, beta_start, beta_end) -> None:
        self.timesteps = timesteps
        # betas : add noise, alphas : denoising
        self.betas = np.linspace(beta_start, beta_end, timesteps) # Linear interpolated array of noise intensity, length is timesteps
        self.alphas = 1 - self.betas                    # The aplha value of each time step is calculated to represent the denoising retention ratio
        self.alphas_bars = np.cumprod(self.alphas)      # the cumlative product represents the total retention ratio from time step 0 to the current time step

    '''
    ### 1. Forward Diffusion Process
    Starting with a distribution of complex data (such as a picture), 
    noise is gradually added so that it becomes closer and closer to a simple known distribution (such as Gaussian distribution)
    '''
    def forward_diffsion(self, x0): # x0: raw picture or signals
        noise = np.random.normal(size=x0.shape)
        xt = x0
        trajectory = [xt]
        for t in range(self.timesteps): 
            xt = np.sqrt(self.alphas[t]) * xt + np.sqrt(self.betas[t]) * noise    # np.sqrt(self.betas[t]) : Intensity of noise
            trajectory.append(xt)
        return trajectory
    
    """
    ### 2. Reverse Diffusion Process
    Learn a reverse process, strarting with a simple 'known distribution', gradually removing noise and returning to a distribution of complex data.
    This process is accomplished by training a neural network that learns how to remove noise at each step
    """
    def reverse_diffusion(self, xt):
        noise = np.random.normal(size=xt.shape)
        for t in reversed(range(self.timesteps)):
            # xt: The result of the last time step of the forward diffusion process
            # np.sqrt(self.betas[t]) * noise : use "-" to noise removal
            # np.sqrt(self.alphas[t]) : use "/" to scale
            xt = (xt - np.sqrt(self.betas[t]) * noise) / np.sqrt(self.alphas[t])
        return xt
    
    '''
    ### 3. Conditional Generation
    In order to achieve generation based on text input, the model not only learns how to remove noise,
    but also learns how to guide the denosing process according to given conditions (such as text descriptions).
    '''
    def generate(self, x0):
        trajectory = self.forward_diffsion(x0)    # The trajectory of all intermediate steps in the diffusion process is obtained
        xt = trajectory[-1]
        x0_hat = self.reverse_diffusion(xt)       # The denoised data x0_hat is obtained
        return x0_hat, trajectory
    
if __name__ == "__main__":
    # generate a simple two-dimemsional data point(e.g., a sine wave)
    x0 = np.linspace(0, 2 * np.pi, 100)
    y0 = np.sin(x0)

    # Initialize and run the diffusion model
    model = SimpleDiffusionModel(timesteps=50, beta_start=0.01, beta_end=0.1)
    x0_hat, trajectory = model.generate(y0)

    # Visualize raw data, noise data and generated data
    plt.figure(figsize=(10, 6))
    plt.plot(x0, y0, label="Original data")
    plt.plot(x0, trajectory[-1], label='Noisy data')
    plt.plot(x0, x0_hat, label='Generated data')
    plt.legend()
    plt.savefig("Jiang_diffusion_model_output.png")
    plt.show()
    
