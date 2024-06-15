import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
import torch.utils.data

class PathPlanningDiffusionModel:
    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_bars = np.cumprod(self.alphas)

    def forward_diffusion(self, x0, goal):
        noise = torch.rand_like(x0)
        xt = x0
        trajectory = [xt]
        for t in range(self.timesteps):
            xt = torch.sqrt(torch.tensor(self.alphas[t])) * xt + torch.sqrt(torch.tensor(self.betas[t])) * noise + (1 - torch.sqrt(torch.tensor(self.alphas_bars[t]))) * goal
            trajectory.append(xt)
        return trajectory

    def reverse_diffusion(self, xt, model, goal):
        goal = goal if goal.dim() == 2 else goal.unsqueeze(0)  # make sure goal is 2 dim
        for t in reversed(range(self.timesteps)):
            xt = xt if xt.dim() == 2 else xt.unsqueeze(0)  # make sure xt is 2 dim

            print(f"xt shape: {xt.shape}, goal shape: {goal.shape}")
            noise_pred = model(torch.cat([xt, goal], dim=-1))
            print(f"noise_pred.shape:{noise_pred.shape}")

            # Ensure noise_pred is correctly sliced to match xt
            noise_pred = noise_pred[:, :xt.shape[1]]

            xt = (xt - torch.sqrt(torch.tensor(self.betas[t])) * noise_pred) / torch.sqrt(torch.tensor(self.alphas[t])) + (1 - torch.sqrt(torch.tensor(self.alphas_bars[t]))) * goal
            xt = xt.squeeze(0)  # After processing, the xt is restored to its original dimensions
        return xt

    def generate(self, x0, model, goal):
        trajectory = self.forward_diffusion(x0, goal)
        xt = trajectory[-1]  # get the last item in trajectory
        print(f"xt shape before reverse: {xt.shape}, goal shape: {goal.shape}")

        reversed_trajectory = []
        for _ in range(self.timesteps):
            xt = self.reverse_diffusion(xt, model, goal)
            reversed_trajectory.append(xt)  # make sure that 'trajectories' are generated only in forward diffusion, and ensure that generated elements are dimensionally consistent
        x0_hat = xt
        return x0_hat, trajectory
    
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)   

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def simulate_data(num_samples, timesteps, input_dim, output_dim):
    model = PathPlanningDiffusionModel(timesteps, 0.01, 0.1)
    data = []
    for _ in range(num_samples):
        x0 = torch.randn(input_dim).unsqueeze(0)
        goal = torch.randn(input_dim).unsqueeze(0)

        print(f"x0 shape: {x0.shape}, goal shape: {goal.shape}")

        trajectory = model.forward_diffusion(x0, goal)

        x0_hat, trajectory = model.generate(x0, lambda x: torch.rand_like(x), goal)

        for t, xt in enumerate(trajectory[:-1]):
            next_xt = trajectory[t + 1] 
            if next_xt.dim() < xt.dim():
                next_xt = next_xt.unsqueeze(0)
            data.append((torch.cat([xt, goal], dim=-1), (xt - next_xt) / np.sqrt(model.betas[t])))
    return data

def apply_constraints(trajectory, goal, lane_width, car_width, car_length):
    constrained_trajectory = []
    for t, point in enumerate(trajectory):
        x, y = point
        # 初始位置位于右侧车道
        if t < len(trajectory) // 3:
            # 保持在右侧车道，不要撞到实线
            y = np.clip(y, car_width / 2, lane_width - car_width / 2)
        else:
            # 变道到左侧车道
            y = np.clip(y, lane_width + car_width / 2, lane_width * 2 - car_width / 2)
        
        # 确保不会撞击目标车
        if np.abs(x - goal[0]) < car_length:
            x = goal[0] + car_length

        constrained_trajectory.append([x, y])
    
    # 平滑路径
    constrained_trajectory = np.array(constrained_trajectory)
    smoothed_trajectory = np.copy(constrained_trajectory)
    
    for i in range(1, len(constrained_trajectory) - 1):
        smoothed_trajectory[i] = (constrained_trajectory[i-1] + constrained_trajectory[i] + constrained_trajectory[i+1]) / 3

    return smoothed_trajectory

if __name__ == "__main__":
    # define hyperparameter
    timesteps = 50
    input_dim = 4
    hidden_dim = 128
    output_dim = 2
    num_samples = 1000
    batch_size = 64
    epochs = 10

    # init model and data
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    data = simulate_data(num_samples, timesteps, input_dim, output_dim)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # train model
    for epoch in range(epochs):
        for x, y in data_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # init diffusion model
    diffusion_model = PathPlanningDiffusionModel(timesteps, 0.01, 0.1)

    # init ego and goal position
    x0 = torch.tensor([0, 3])
    goal = torch.tensor([14, 3])

    # genrative path
    x0 = x0.unsqueeze(0)
    goal = goal.unsqueeze(0)
    x0_hat, trajectory = diffusion_model.generate(x0, model, goal)

    # add constrain
    constrained_trajectory = apply_constraints(trajectory, goal.squeeze(0), 4, 1, 2)

    # plot path
    filenames = []
    for i, point in enumerate(constrained_trajectory):
        plt.figure()
        ax = plt.gca()

        # 绘制车道线和虚线
        plt.plot([0, 0], [-1, 26], 'k-')
        plt.plot([4, 4], [-1, 26], 'k-')
        plt.plot([8, 8], [-1, 26], 'k-')
        plt.plot([2, 2], [-1, 26], 'k:')
        plt.plot([6, 6], [-1, 26], 'k:')

        # 绘制自车和目标车位置
        car = plt.Rectangle((point[1] - 0.5, point[0] - 1), 1, 2, fill=True, color='blue', label='Ego')
        goal_rect = plt.Rectangle((goal[1] - 0.5, goal[0] - 1), 1, 2, fill=True, color='red', label='Goal')
        ax.add_patch(car)
        ax.add_patch(goal_rect)

        # 绘制路径
        if i > 0:
            path = np.array(constrained_trajectory[:i + 1])
            plt.plot(path[:, 1], path[:, 0], 'g-', label='Path')

        plt.xlim(-1, 6)
        plt.ylim(-1, 30)
        plt.legend(loc='upper left')
        plt.title(f'Time step {i}')

        filename = f'frame_{i}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()
    

    # 保存所有帧为动画
    with imageio.get_writer('path_planning.gif', mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # 清除生成的帧文件
    import os
    for filename in filenames:
        os.remove(filename)

    print("GIF生成完毕")
