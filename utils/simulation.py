from . import plotting_utils
import numpy as np
from matplotlib import pyplot, animation
class BotSimulation:
    def __init__(self, lidar_func, robot_size, odometry_noise, lidar_noise, velocities, num_scans=100):
        self.cast_lidar_rays = lidar_func
        self.size = robot_size
        self.odometry_noise = odometry_noise
        self.lidar_noise = lidar_noise
        self.num_scans = num_scans
        self.path = self.simulate_robot_movement(
            initial_x=8, initial_y=8, initial_orientation=45, wheel_velocities=velocities, dt=2, noise=0.0, scale=1.0)
        self.noise_path = self.simulate_robot_movement(
            initial_x=8, initial_y=8, initial_orientation=45, wheel_velocities=velocities, dt=2, noise=odometry_noise, scale=1.0)
        self.map_objects = plotting_utils.get_map_objects(robot_size*10)
        self.setup_fig()
    def setup_fig(self):
        self.fig, self.ax = plotting_utils.generate_map(self.map_objects, size=self.size * 10)
        self.ax.set_xlim(-self.size*10, self.size * 10 * 2 + self.size*10)
        self.ax.set_ylim(-self.size*10, self.size * 10 + self.size*10)
        self.ax.set_aspect('equal')
        self.all_data = self.run_simulation()

    def wheel_velocities_to_linear_angular(self, v_left, v_right, wheel_radius=.75, wheel_base=4, noise=0.0,scale = 1):
        v_left = v_left * scale
        v_right = v_right * scale
        v_left += np.random.normal(0, noise)
        v_right += np.random.normal(0, noise)
        # Calculate linear and angular velocities, assuming wheels are on either side of the robot
        # factor in that one wheel not moving will slow down that side and cause the robot to turn
        linear_velocity = (v_left + v_right) / 2
        angular_velocity = (v_right - v_left) / wheel_base
        # Convert to linear velocity in m/s
        linear_velocity = linear_velocity * wheel_radius
        angular_velocity = angular_velocity * wheel_radius
        # Convert angular velocity to degrees per second
        angular_velocity = np.degrees(angular_velocity)

        return linear_velocity, angular_velocity
    # function to take linear and angular velocities and return the new position and orientation of the robot
    def update_robot_position(self, x, y, orientation, linear_velocity, angular_velocity, dt=2):
        # Update the position and orientation of the robot
        x += linear_velocity * np.cos(np.radians(orientation)) * dt
        y += linear_velocity * np.sin(np.radians(orientation)) * dt
        orientation += angular_velocity * dt

        return x, y, orientation

    def simulate_robot_movement(self, initial_x, initial_y, initial_orientation, wheel_velocities, dt=2, noise=0.0, scale=1.0):
        x, y, orientation = initial_x, initial_y, initial_orientation
        trajectory = [(x, y, orientation)]

        for v_left, v_right in wheel_velocities:
            linear_velocity, angular_velocity = self.wheel_velocities_to_linear_angular(v_left, v_right, noise=noise, scale=scale)
            x, y, orientation = self.update_robot_position(x, y, orientation, linear_velocity, angular_velocity, dt)
            trajectory.append((x, y, orientation))

        return trajectory

    def relocate_scan(self, data):
        """
        Visualizes what the robot perceives. It takes the local scan
        and transforms it back to the global frame using the robot's noisy pose estimate.
        This shows us the map the robot is building.
        """
        # Unpack the data we need
        _, _, _, pred_lidar, _, local_scan_points = data

        # Get the robot's estimated (noisy) pose
        robot_x, robot_y = pred_lidar[0]
        front_x, front_y = pred_lidar[1]

        # Calculate the robot's estimated orientation (theta)
        robot_theta = np.arctan2(front_y - robot_y, front_x - robot_x)

        # Pre-calculate cosine and sine for the forward rotation (+theta)
        cos_theta = np.cos(robot_theta)
        sin_theta = np.sin(robot_theta)

        relocated_scan = []
        for local_point in local_scan_points:
            local_x, local_y = local_point

            # Step 1: Rotate the local point to align with the global frame
            rotated_x = local_x * cos_theta - local_y * sin_theta
            rotated_y = local_x * sin_theta + local_y * cos_theta

            # Step 2: Translate the rotated point to the robot's global position
            global_x = rotated_x + robot_x
            global_y = rotated_y + robot_y

            relocated_scan.append((global_x, global_y))

        return relocated_scan

    def generate_timestep(self, step):
        # Escape is if the step is out of bounds
        if step >= len(self.path):
            return None

        # Get the true robot position and orientation
        x, y, orientation = self.path[step]
        # Create the true robot and lidar objects
        # These are the true values without noise
        true_robot, true_lidar = plotting_utils.create_robot(x, y, orientation, size=self.size)
        # Cast clean lidar rays from the true lidar position and check for intersections with walls and circles
        global_intersections = self.cast_lidar_rays(true_lidar, self.map_objects, num_rays=self.num_scans)

        # Get the predicted robot position and orientation with noise
        # This is the noisy (incorrect) version of the robot's position and orientation
        noise_x, noise_y, noise_orientation = self.noise_path[step]
        # Create the predicted robot and lidar objects
        pred_robot, pred_lidar = plotting_utils.create_robot(noise_x, noise_y, noise_orientation, size=self.size)

        # Get the true lidar position and orientation
        true_x, true_y = true_lidar[0]
        front_x, front_y = true_lidar[1]
        # Calculate the true lidar orientation (theta)
        true_theta = np.arctan2(front_y - true_y, front_x - true_x)
        cos_inv_theta = np.cos(-true_theta)
        sin_inv_theta = np.sin(-true_theta)

        # Here we will add noise to the lidar intersections and transform them to the global positions given by the local frame of the noisy robot
        local_intersections = []
        for point_global in global_intersections:
            dx = point_global[0] - true_x
            dy = point_global[1] - true_y
            local_x = dx * cos_inv_theta - dy * sin_inv_theta
            local_y = dx * sin_inv_theta + dy * cos_inv_theta

            # Add sensor noise to the local points and add them to the list
            noisy_local_x = local_x + np.random.normal(0, self.lidar_noise)
            noisy_local_y = local_y + np.random.normal(0, self.lidar_noise)
            local_intersections.append((noisy_local_x, noisy_local_y))

        return true_robot, pred_robot, true_lidar, pred_lidar, global_intersections, local_intersections

    def run_simulation(self):
        all_data= []
         # Iterate through the path and generate data for each timestep
        for step in range(len(self.path)):
            timestep_data = self.generate_timestep(step)
            if timestep_data is None:
                break
            all_data.append(timestep_data)
        return all_data  # Unzip the data into separate lists for true and predicted values

    def animate_simulation_and_noise(self, all_data):

        # Create placeholders for the robot and lidar lines
        true_robot_line, = self.ax.plot([], [], color='blue', label='True Robot')
        pred_robot_line, = self.ax.plot([], [], color='orange', label='Predicted Robot')
        true_lidar_line, = self.ax.plot([], [], color='green', label='True Lidar')
        pred_lidar_line, = self.ax.plot([], [], color='red', label='Predicted Lidar')
        true_intersection_lines = []
        simul_intersection_lines = []
        for _ in range(len(all_data[0][4])):
            true_intersection_lines.append(self.ax.plot([],[], 'go-')[0])
        for _ in range(len(all_data[0][5])):
            simul_intersection_lines.append(self.ax.plot([],[], 'ro-')[0])

        # Extract the true and predicted paths for plotting
        # The paths are lists of tuples (x, y) because the dotted lign has no orientation
        true_path = np.array(self.path)[:, :2]
        pred_path = np.array(self.noise_path)[:, :2]
        # Plot the true and predicted paths
        self.ax.plot(true_path[:, 0], true_path[:, 1], color='blue', linewidth=.4, marker='8', markersize=3, alpha=0.2, label='True Path')
        self.ax.plot(pred_path[:, 0], pred_path[:, 1], color='orange', linewidth=.4, marker='8', markersize=3, alpha=0.2, label='Predicted Path')
        self.ax.legend()
        def init():
            # Initialize empty data for the robot and lidar
            true_robot_line.set_data([], [])
            pred_robot_line.set_data([], [])
            true_lidar_line.set_data([], [])
            pred_lidar_line.set_data([], [])
            for line in true_intersection_lines:
                line.set_data([], [])
            for line in simul_intersection_lines:
                line.set_data([], [])
            return (true_robot_line, pred_robot_line, true_lidar_line,
                    pred_lidar_line, true_intersection_lines, simul_intersection_lines)
        def update(frame):
            # Get the current true and predicted robot and lidar data
            true_robot, pred_robot, true_lidar, pred_lidar, true_intersection, _ = all_data[frame]
            # Update the robot and lidar line data
            true_robot_line.set_data(*zip(*true_robot))
            pred_robot_line.set_data(*zip(*pred_robot))
            true_lidar_line.set_data(*zip(*true_lidar))
            pred_lidar_line.set_data(*zip(*pred_lidar))
            # Update the intersection lines
            for i, intersection in enumerate(true_intersection):
                true_intersection_lines[i].set_data((true_lidar[1][0], intersection[0]), (true_lidar[1][1], intersection[1]))
            # plot the predicted intersections coming from the predicted lidar, transposed to adjust for the noised fov
            new_intersections = self.relocate_scan(data=all_data[frame])
            for i, intersection in enumerate(new_intersections):
                simul_intersection_lines[i].set_data((pred_lidar[1][0], intersection[0]), (pred_lidar[1][1], intersection[1]))
            return (true_robot_line, pred_robot_line, true_lidar_line,
                    pred_lidar_line, true_intersection_lines, simul_intersection_lines)
        # Create the animation
        # this function is a bit complex, it updates the robot and lidar lines for each frame
        # by using a generator init function and an update function
        # The init function initializes the lines and the update function updates the lines for each frame
        # The blit=False argument is used to redraw the entire figure for each frame
        # This is necessary because we are updating the data of the lines, not just their visibility
        anim = animation.FuncAnimation(
            self.ax.figure, update, frames=len(self.path), init_func=init, blit=False)
        return anim

def plot_predicted_map(simulation):
    pyplot.clf()
    ax1 = pyplot.subplot(121)
    ax2 = pyplot.subplot(122)
    ax1.set_title('Predicted Map')
    ax2.set_title('True Map')
    ax1.set_xlim(-simulation.size*10, simulation.size*10*2 + simulation.size*10)
    ax1.set_ylim(-simulation.size*10, simulation.size*10 + simulation.size*10)
    ax2.set_xlim(-simulation.size*10, simulation.size*10*2 + simulation.size*10)
    ax2.set_ylim(-simulation.size*10, simulation.size*10 + simulation.size*10)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    for step in range(len(simulation.all_data)):
        _, _, _, _, true, _ = simulation.all_data[step]
        noise = simulation.relocate_scan(data=simulation.all_data[step])
        for intersection in noise:
            ax1.plot(intersection[0], intersection[1], 'ro', markersize=1, alpha=0.5)
        for intersection in true:
            ax2.plot(intersection[0], intersection[1], 'go', markersize=1, alpha=0.5)
    pyplot.show()


# ============== Old lidar testing notebook code ==============
# This is a new cell to replace the old ICP test cell (id: 6148785c)

# --- ICP Alignment Verification Cell ---

# # 1. SETUP: Choose two scans to compare
# pose_graph = PoseGraph(simulation, 1e-6, 1000)
# first_idx = 50
# second_idx = 120

# # 2. DATA: Get the raw LOCAL scans and the TRUE global poses
# local_scan_1 = torch.tensor(simulation.all_data[first_idx][5], dtype=torch.float32)
# local_scan_2 = torch.tensor(simulation.all_data[second_idx][5], dtype=torch.float32)

# true_pose_1_list = simulation.path[first_idx]
# true_pose_2_list = simulation.path[second_idx]

# true_pose_1 = torch.tensor([true_pose_1_list[0], true_pose_1_list[1], np.deg2rad(true_pose_1_list[2])], dtype=torch.float32)
# true_pose_2 = torch.tensor([true_pose_2_list[0], true_pose_2_list[1], np.deg2rad(true_pose_2_list[2])], dtype=torch.float32)

# # 3. ICP: Find the transformation from scan 2 to scan 1
# noisy_pose_1 = pose_graph.nodes[first_idx]
# noisy_pose_2 = pose_graph.nodes[second_idx]
# relative_transform_guess = pose_graph.compute_relative_transform(noisy_pose_2, noisy_pose_1)

# dx, dy, dtheta = relative_transform_guess
# cos_dt, sin_dt = torch.cos(dtheta), torch.sin(dtheta)
# initial_guess = torch.eye(3).unsqueeze(0)
# initial_guess[:, 0, 0], initial_guess[:, 0, 1] = cos_dt, -sin_dt
# initial_guess[:, 1, 0], initial_guess[:, 1, 1] = sin_dt, cos_dt
# initial_guess[:, 0, 2], initial_guess[:, 1, 2] = dx, dy

# _, icp_transform, alignment_errors, _ = pose_graph.batch_trimmed_icp(
#     local_scan_2.unsqueeze(0), local_scan_1.unsqueeze(0),
#     initial_guess=initial_guess.to(dtype=torch.float32),
#     max_iterations=1000, tolerance=1e-6
# )
# icp_transform = icp_transform.squeeze(0)
# print(f"ICP Alignment Error: {alignment_errors.item()}")

# # 4. VISUALIZATION HELPERS
# def to_global(local_points, global_pose):
#     x, y, theta = global_pose
#     cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
#     R = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
#     return (torch.einsum('ij,nj->ni', R, local_points) + torch.tensor([x, y])).numpy()

# def apply_transform(points, transform_mat):
#     R, t = transform_mat[:2, :2], transform_mat[:2, 2]
#     return torch.einsum('ij,nj->ni', R, points) + t

# # --- FIX: Helper function to draw the map on a given axis ---
# def draw_map_on_ax(ax, room_objects, map_size):
#     """Draws the walls and circles on a provided matplotlib axis."""
#     for wall in room_objects['walls']:
#         ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black')
#     for circle in room_objects['circles']:
#         ax.add_patch(pyplot.Circle(circle['center'], circle['radius'], color='black', fill=False))
#     ax.set_xlim(0, map_size)
#     ax.set_ylim(0, map_size)
#     ax.set_aspect('equal', adjustable='box')

# # --- PLOTTING ---
# pyplot.clf()
# fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(16, 8))

# # Draw the map on both subplots
# draw_map_on_ax(ax1, simulation.map_objects, simulation.size * 10)
# draw_map_on_ax(ax2, simulation.map_objects, simulation.size * 10)

# # Plot 1: Before ICP
# global_scan_1_true = to_global(local_scan_1, true_pose_1)
# global_scan_2_true = to_global(local_scan_2, true_pose_2)
# ax1.scatter(global_scan_1_true[:, 0], global_scan_1_true[:, 1], color='blue', s=5, label=f'Scan {first_idx} (True Pose)')
# ax1.scatter(global_scan_2_true[:, 0], global_scan_2_true[:, 1], color='orange', s=5, label=f'Scan {second_idx} (True Pose)')
# ax1.set_title('Scans at True Positions (Before ICP)')
# ax1.set_xlim(-simulation.size * 10, simulation.size * 10 * 2 + simulation.size * 10)
# ax1.set_ylim(-simulation.size * 10, simulation.size * 10 + simulation.size * 10)
# ax1.legend()

# # Plot 2: After ICP
# transformed_scan_2 = apply_transform(local_scan_2, icp_transform)
# global_scan_1_aligned = to_global(local_scan_1, true_pose_1)
# global_scan_2_aligned = to_global(transformed_scan_2, true_pose_1)
# ax2.scatter(global_scan_1_aligned[:, 0], global_scan_1_aligned[:, 1], color='blue', s=5, label=f'Scan {first_idx}')
# ax2.scatter(global_scan_2_aligned[:, 0], global_scan_2_aligned[:, 1], color='orange', s=5, alpha=0.7, label=f'Scan {second_idx} (Aligned)')
# ax2.set_title('Scans Aligned by ICP')
# ax2.set_xlim(-simulation.size * 10, simulation.size * 10 * 2 + simulation.size * 10)
# ax2.set_ylim(-simulation.size * 10, simulation.size * 10 + simulation.size * 10)
# ax2.legend()

# pyplot.show()
# ======================================================