import numpy as np
import matplotlib.pyplot as plt
import imageio

class PathPlanningDiffusionModel:
    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_bars = np.cumprod(self.alphas)

    def forward_diffusion(self, x0, goal):
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
            xt = (xt - np.sqrt(self.betas[t]) * noise) / np.sqrt(self.alphas[t]) + (1 - np.sqrt(self.alphas_bars[t])) * goal
        return xt

    def generate(self, x0, goal):
        trajectory = self.forward_diffusion(x0, goal)
        xt = trajectory[-1]
        x0_hat = self.reverse_diffusion(xt, goal)
        return x0_hat, trajectory


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
    # 设置参数
    timesteps = 50
    beta_start = 0.01
    beta_end = 0.1
    lane_width = 4
    car_width = 1
    car_length = 2

    # 初始化扩散模型
    model = PathPlanningDiffusionModel(timesteps, beta_start, beta_end)

    # 初始化自车和目标车位置
    x0 = np.array([0, 3])
    goal = np.array([14, 3])

    # 生成路径
    x0_hat, trajectory = model.generate(x0, goal)

    # 施加约束条件
    constrained_trajectory = apply_constraints(trajectory, goal, lane_width, car_width, car_length)

    # 创建GIF动画
    filenames = []
    for i, point in enumerate(constrained_trajectory):
        plt.figure()
        ax = plt.gca()

        # 绘制车道线和虚线
        plt.plot([0, 0], [-1, 26], 'k-')
        plt.plot([lane_width, lane_width], [-1, 26], 'k-')
        plt.plot([lane_width * 2, lane_width * 2], [-1, 26], 'k-')
        plt.plot([lane_width / 2, lane_width / 2], [-1, 26], 'k:')
        plt.plot([lane_width + lane_width / 2, lane_width + lane_width / 2], [-1, 26], 'k:')

        # 绘制自车和目标车位置
        car = plt.Rectangle((point[1] - car_width / 2, point[0] - car_length / 2), car_width, car_length, fill=True, color='blue', label='Ego')
        goal_rect = plt.Rectangle((goal[1] - car_width / 2, goal[0] - car_length / 2), car_width, car_length, fill=True, color='red', label='Goal')
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
