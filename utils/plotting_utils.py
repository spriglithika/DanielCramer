from matplotlib import pyplot
from matplotlib import animation
import numpy as np
# def generate_map(size = 10):
#     fig, ax = pyplot.subplots()
#     height = size
#     width = size * 2
#     ax.set_xlim(0, height)
#     ax.set_ylim(0, height)

#     # Draw the walls
#     ax.plot([0, width], [0, 0], color='black')  # Bottom wall
#     ax.plot([0, width], [height, height], color='black')  # Top wall
#     ax.plot([0, 0], [0, height], color='black')  # Left wall
#     ax.plot([width, width], [0, height], color='black')  # Right wall

#     # Draw the circles
#     r = height/10
#     circle1 = pyplot.Circle((r * 2, height - r * 2), radius=r, color='black', fill=True)
#     circle2 = pyplot.Circle((width - r * 2, r * 2), radius=r, color='black', fill=True)

#     ax.add_artist(circle1)
#     ax.add_artist(circle2)

#     # Draw the wall in the middle
#     ax.plot([height, height], [0, height / 2], color='black')

#     return fig, ax

# def get_map_objects(size=10):
#     height = size
#     width = size * 2
#     radius = height / 10
#     objects = {
#         'walls': [
#             [(0, 0), (width, 0)],
#             [(0, height), (width, height)],
#             [(0, 0), (0, height)],
#             [(width, 0), (width, height)],
#             [(height, 0), (height, height / 2)]
#         ],
#         'circles': [
#             {'center': (radius * 2, height - radius * 2), 'radius': radius},
#             {'center': (width - radius * 2, radius * 2), 'radius': radius}
#         ]
#     }
#     return objects

def generate_map(objects,size =10):
    fig, ax = pyplot.subplots()
    ax.set_xlim(0, size * 2)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    # Draw the walls
    for wall in objects['walls']:
        ax.plot(*zip(*wall), color='black')
    # Draw the circles
    for circle in objects['circles']:
        circle_artist = pyplot.Circle(circle['center'], radius=circle['radius'], color='black', fill=True)
        ax.add_artist(circle_artist)
    return fig, ax

def get_map_objects(size=10):
    height = size
    width = size * 2
    radius = height / 10
    objects = {
        'walls': [
            [(0, 0), (width, 0)],
            [(0, height), (width, height)],
            [(0, 0), (0, height)],
            [(width, 0), (width, height)],
            [(.6*height, 0), (height, height / 2)],
            [(1.1*height, 0), (height, height / 2)],
            [(1.8*height, height), (width, .8*height)],
            [(.8*height, height), (.8*height, .9*height)],
            [(1.2*height, height), (1.2*height, .9*height)],
            [(.8*height, .9*height), (1.2*height, .9*height)],
            [(.1*height, 0), (.1*height, .05*height)],
            [(.1*height, .05*height), (.05*height, .1*height)],
            [(0, .1*height), (.05*height, .1*height)],

        ],
        'circles': [
            {'center': (radius * 2, height - radius * 2), 'radius': radius*1.3},
            {'center': (width - radius * 2, radius * 2), 'radius': radius}
        ]
    }
    return objects

def create_robot(x, y, orientation, size=1):
    # Create a simple octagonal robot with a front line
    # Define the octagon vertices
    angle = orientation * (3.14159 / 180)  # Convert degrees to radians
    octagon = [
        (x + size * 0.707 * np.cos(angle + i * 3.14159 / 4), y + size * 0.707 * np.sin(angle + i * 3.14159 / 4))
        for i in range(8)
    ]
    octagon.append(octagon[0])  # Close the shape

    front_x = x + size * .626 * np.cos(angle)
    front_y = y + size * .626 * np.sin(angle)
    lidar = [(x, y), (front_x, front_y)]
    return octagon, lidar

def plot_robot(ax, robot, lidar):
    # Plot the octagon
    robot = np.array(robot)
    ax.plot(robot[:, 0], robot[:, 1], color='blue')
    # Plot the front line
    ax.plot(*zip(*lidar), color='red', linewidth=2)

def show_room_and_robot(ax, room, robot, lidar):
    # Draw the walls
    for wall in room['walls']:
        ax.plot(*zip(*wall), color='black')
    # Draw the circles
    for circle in room['circles']:
        circle_artist = pyplot.Circle(circle['center'], radius=circle['radius'], color='black', fill=True)
        ax.add_artist(circle_artist)
    plot_robot(ax, robot, lidar)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 40)
    # Set the aspect ratio to be equal
    ax.set_aspect('equal')

def animate_robot_movement(ax, path, size=1):
    # Create placeholders for the robot and lidar line
    robot_line, = ax.plot([], [], color='blue')  # Octagon
    lidar_line, = ax.plot([], [], color='red', linewidth=2)  # Front line

    def init():
        # Initialize empty data for the robot and lidar
        robot_line.set_data([], [])
        lidar_line.set_data([], [])
        return robot_line, lidar_line

    def update(frame):
        # Get the current position and orientation
        x, y, orientation = path[frame]

        # Define the octagon vertices
        robot, lidar = create_robot(x, y, orientation, size=size)
        robot = np.array(robot)

        # Update the robot and lidar line data
        robot_line.set_data(robot[:, 0], robot[:, 1])
        lidar_line.set_data(*zip(*lidar))
        return robot_line, lidar_line

    # Create the animation
    anim = animation.FuncAnimation(
        ax.figure, update, frames=len(path), init_func=init, blit=False, interval=400
    )
    return anim

# plot the robot, map, and lidar intersections
def plot_lidar_intersections(ax, robot, lidar, lidar_intersections):
    # Plot the robot
    plot_robot(ax, robot, lidar)

    # Plot the lidar intersections
    for intersection in lidar_intersections:
        ax.plot((lidar[1][0], intersection[0]), (lidar[1][1], intersection[1]), 'ro-')  # Red dot for intersection

    # Set limits and aspect ratio
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 40)
    ax.set_aspect('equal')

if __name__ == "__main__":
    # Example usage
    objects = get_map_objects(40)
    fig, ax = generate_map(objects, 40)

    # Create a robot at position (10, 20) with orientation 45 degrees
    robot, lidar = create_robot(10, 10, 45, 4)

    # Plot the room and robot
    show_room_and_robot(ax, objects, robot, lidar)

    # Example path for animation
    # path = [(10, 20, 45), (15, 25, 90), (20, 30, 135)]

    # Animate the robot movement
    # anim = animate_robot_movement(ax, path)

    pyplot.show()