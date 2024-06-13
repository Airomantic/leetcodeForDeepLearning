import numpy as np

import matplotlib.pyplot as plt


class SimpleDiffusionModel:
    def __init__(self, timesteps, beta_start, beta_end) -> None:
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_bars = np.cumprod(self.alphas)

    def forward_diffsion(self, x0):
        noise = np.random.normal(size=x0.shape)
        xt = x0
        trajectory = [xt]
        for t in range(self.timesteps):
            xt = np.sqrt(self.alphas[t]) * xt + np.sqrt(self.betas[t]) * noise
            trajectory.append(xt)
        return trajectory
    
    def reverse_diffusion(self, xt):
        noise = np.random.normal(size=xt.shape)
        for t in reversed(range(self.timesteps)):
            xt = (xt - np.sqrt(self.betas[t]) * noise) / np.sqrt(self.alphas[t])
        return xt
    
    def generate(self, x0):
        trajectory = self.forward_diffsion(x0)
        xt = trajectory[-1]
        x0_hat = self.reverse_diffusion(xt)
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
    
