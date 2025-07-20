import numpy as np
def wall_intersection(wall, p1, p2):
    p3 = wall[0]
    p4 = wall[1]
    """Find intersection point of two lines defined by points p1, p2 and p3, p4."""
    denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    if denom == 0:
        return None  # Lines are parallel

    ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom
    ub = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / denom

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return (p1[0] + ua * (p2[0] - p1[0]), p1[1] + ua * (p2[1] - p1[1]))
    return None
# Function to find intersection points between a line and a circle
def circle_intersection(circle, p1, p2):
    """Find intersection points between a line defined by points p1, p2 and a circle defined by center and radius."""
    center = circle['center']
    radius = circle['radius']
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - center[0]
    fy = p1[1] - center[1]

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []  # No intersection

    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b + sqrt_discriminant) / (2 * a)
    t2 = (-b - sqrt_discriminant) / (2 * a)

    intersections = []
    if 0 <= t1 <= 1:
        intersections.append((p1[0] + t1 * dx, p1[1] + t1 * dy))
    if 0 <= t2 <= 1:
        intersections.append((p1[0] + t2 * dx, p1[1] + t2 * dy))

    return intersections

wheel_velocities = [
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0), # forward
    (2.0, 1.0),
    (2.0, 1.0), # soft turn right
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0), # forward
    (2.0, 1.0),
    (2.0, 1.0), # soft turn right
    (2.0, 2.0),
    (2.0, 2.0), # forward
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0), # wide left turn
    (1.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0), # sharper left turn
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0), # wide left turn
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0), # forward
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0), # slow turn left
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0),
    (1.5, 2.0), # wide left turn
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.5, 1.0), # slow turn left
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0), # straight around the ttriangle
    (2.0, 1.0),
    (2.0, 1.0),
    (2.0, 0.5),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 0.5),
    (2.0, 0.5), # turn around the triangle
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (2.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (1.0, -1.0),
    (2.0, 2.0),
    (2.0, 2.0),
    (1.0, -1.0),
    (2.0, 2.0),
    (1.0, -1.0),
    (1.0, 0.0),
    (1.0, 0.0),
    (2.0, 2.0),
    (1.0, 0.0),
    (2.0, 2.0),
]
