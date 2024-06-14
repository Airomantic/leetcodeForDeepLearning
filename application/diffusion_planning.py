import numpy as np
import matplotlib.pyplot as plt
import imageio

class PathPlanningDiffusionModel:
    def __init__(self, timesteps, beta_start, beta_end) -> None:
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_bars = np.cumprod(self.alphas)

    def forward_diffsion(self, x0, goal): # x0: raw picture or signals
        noise = np.random.normal(size=x0.shape)
        xt = x0
        trajectory = [xt]
        for t in range(self.timesteps): 
            xt = np.sqrt(self.alphas[t]) * xt + np.sqrt(self.betas[t]) * noise + (1 - np.sqrt(self.alphas_bars[t])) * goal
            trajectory.append(xt)
        return trajectory
    
    def reverse_diffusion(self, xt, goal):
        noise = np.random.normal(size=xt.shape)
        for t in reversed(range(self.timesteps)):
            # xt: The result of the last time step of the forward diffusion process
            # np.sqrt(self.betas[t]) * noise : use "-" to noise removal
            # np.sqrt(self.alphas[t]) : use "/" to scale
            xt = (xt - np.sqrt(self.betas[t]) * noise) / np.sqrt(self.alphas[t]) + (1 - np.sqrt(self.alphas_bars[t])) * goal
        return xt
    
    def generate(self, x0, goal):
        trajectory = self.forward_diffsion(x0, goal)    # The trajectory of all intermediate steps in the diffusion process is obtained
        xt = trajectory[-1]
        x0_hat = self.reverse_diffusion(xt, goal)       # The denoised data x0_hat is obtained
        return x0_hat, trajectory
    

if __name__ == "__main__":
    # set prameter
    timesteps = 50
    beta_start = 0.01
    beta_end = 0.1

    # init diffusion model
    model = PathPlanningDiffusionModel(timesteps, beta_start, beta_end)

    # init position and goal position
    x0 = np.array([0, 0])
    goal = np.array([10, 5])
    # car outline
    car_width = 1
    car_height = 2

    # genrative path
    x0_hat, trajectory = model.generate(x0, goal)

    # creat GIFs
    filenames = []
    for i, point in enumerate(trajectory):
        plt.figure()
        ax = plt.gca()

        # plot position of ego and goal
        car = plt.Rectangle((point[0] - car_width/2, point[1] - car_height/2), car_width, car_height, fill=True, color='blue', label='Car')
        goal_rect = plt.Rectangle((goal[0] - car_width/2, goal[1] - car_height/2), car_width, car_height, fill=True, color='red', label='goal')
        ax.add_patch(car)
        ax.add_patch(goal_rect)

        # plot path
        if i > 0:
            path = np.array(trajectory[: i + 1])
            plt.plot(path[:, 0], path[:, 1], 'g-', label='Path')

        # plt.plot(point[0], point[1], 'bo') # ego
        # plt.plot(goal[0], goal[1], 'ro') # goal position
        # plt.plot([p[0] for p in trajectory[: i + 1]], [p[1] for p in trajectory[: i + 1]], 'g-')    # path
        # plt.xlim(-1, 11)
        # plt.ylim(-1, 6)
        # filename = f'fram_{i}.png'
        # filenames.append(filename)
        # plt.savefig(filename)
        # plt.close()
        plt.xlim(-1, 11)
        plt.ylim(-1, 6)
        plt.legend(loc='upper left')
        plt.title(f'Time step {i}')

        filename = f'frame_{i}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # Save all frames as animation
    with imageio.get_writer('path_planning.gif', mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # clear genrative frames files
    import os
    for filename in filenames:
        os.remove(filename)

    print("GIF already genrative")