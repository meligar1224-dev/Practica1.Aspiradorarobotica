import HAL
import WebGUI
import Frequency
import cv2
import numpy as np
import math
from collections import deque

MAP_URL = '/resources/exercises/vacuum_cleaner_loc/images/mapgrannyannie.png'
GRID_CELL_SIZE = 15  

map_img = WebGUI.getMap(MAP_URL)

if map_img is None or map_img.size == 0:
    height, width = 800, 800
    gray_map = np.full((height, width), 255, dtype=np.uint8)
else:
    if len(map_img.shape) == 3:
        gray_map = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_map = map_img.copy()
    height, width = gray_map.shape[:2]

X1, Y1, PX1, PY1 = -1.0, 1.5, 680, 560
X2, Y2, PX2, PY2 = 5.096, 5.555, 58, 982

scale_x = abs((PX2 - PX1) / (X2 - X1))
scale_y = abs((PY2 - PY1) / (Y2 - Y1))
SCALE = (scale_x + scale_y) / 2.0

OFFSET_X = PX1 + (X1 * SCALE)
OFFSET_Y = PY1 - (Y1 * SCALE)

def gazebo_to_map(x, y):
    px = int(OFFSET_X - (x * SCALE))
    py = int(OFFSET_Y + (y * SCALE))
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))

def map_to_gazebo(px, py):
    x = (OFFSET_X - px) / SCALE
    y = (py - OFFSET_Y) / SCALE
    return x, y

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

_, binary_map = cv2.threshold(gray_map, 200, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
eroded_map = cv2.erode(binary_map, kernel, iterations=1)

grid_rows = height // GRID_CELL_SIZE
grid_cols = width // GRID_CELL_SIZE

grid_state = np.zeros((grid_rows, grid_cols), dtype=int)

for r in range(grid_rows):
    for c in range(grid_cols):
        cell = eroded_map[r*GRID_CELL_SIZE:(r+1)*GRID_CELL_SIZE, c*GRID_CELL_SIZE:(c+1)*GRID_CELL_SIZE]
        if cell.size > 0 and np.mean(cell) < 60:
            grid_state[r, c] = 1

def get_grid_pos(x, y):
    px, py = gazebo_to_map(x, y)
    c = max(0, min(grid_cols - 1, int(px // GRID_CELL_SIZE)))
    r = max(0, min(grid_rows - 1, int(py // GRID_CELL_SIZE)))
    return r, c

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def get_neighbors(r, c):
    neighbors = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
            neighbors.append((nr, nc))
    return neighbors

def find_path_bfs(start, target):
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        curr = path[-1]
        if curr == target:
            return path
        for nr, nc in get_neighbors(curr[0], curr[1]):
            if grid_state[nr, nc] != 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(path + [(nr, nc)])
    return []

print("Calculando ruta Boustrophedon (BSA)...")

initial_pose = None
while initial_pose is None:
    initial_pose = HAL.getPose3d()
    Frequency.tick(20)

start_r, start_c = get_grid_pos(initial_pose.x, initial_pose.y)
grid_state[start_r, start_c] = 0

visited_sim = np.zeros_like(grid_state)
planned_route = []
return_points = []

current = (start_r, start_c)
while current is not None:
    visited_sim[current] = 1
    if not planned_route or planned_route[-1] != current:
        planned_route.append(current)

    disponibles = [p for p in get_neighbors(current[0], current[1]) if grid_state[p] == 0 and visited_sim[p] == 0]

    if disponibles:
        siguiente = disponibles[0]
        current = siguiente
        continue

    libres = np.argwhere((grid_state == 0) & (visited_sim == 0))
    if len(libres) == 0:
        break

    return_points.append(current)
    candidatos = [(int(p[0]), int(p[1])) for p in libres]
    candidatos.sort(key=lambda p: abs(p[0] - current[0]) + abs(p[1] - current[1]))
    destino = candidatos[0]

    camino_bfs = find_path_bfs(current, destino)
    if not camino_bfs:
        break

    for p in camino_bfs[1:]:
        if not planned_route or planned_route[-1] != p:
            planned_route.append(p)
            visited_sim[p] = 1

    current = destino

print(f"Ruta lista con {len(planned_route)} celdas.")

def render_gui(robot_r, robot_c, route_index):
    # Imagen BGR de color (Fondo gris claro/blanquecino)
    gui_matrix = np.full((height, width, 3), [245, 245, 245], dtype=np.uint8)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * GRID_CELL_SIZE, min(height, (r + 1) * GRID_CELL_SIZE)
            x1, x2 = c * GRID_CELL_SIZE, min(width, (c + 1) * GRID_CELL_SIZE)

            if grid_state[r, c] == 1:
                gui_matrix[y1:y2, x1:x2] = [40, 40, 40]       # Obstáculos (Gris oscuro)
            elif grid_state[r, c] == 2:
                gui_matrix[y1:y2, x1:x2] = [210, 240, 210]   # Celdas visitadas (Verde suave)

    for rr, rc in return_points:
        y1, y2 = rr * GRID_CELL_SIZE, min(height, (rr + 1) * GRID_CELL_SIZE)
        x1, x2 = rc * GRID_CELL_SIZE, min(width, (rc + 1) * GRID_CELL_SIZE)
        gui_matrix[y1:y2, x1:x2] = [0, 140, 255]       # Puntos de retorno (Naranja)

    for r, c in planned_route[route_index:]:
        y1, y2 = r * GRID_CELL_SIZE, min(height, (r + 1) * GRID_CELL_SIZE)
        x1, x2 = c * GRID_CELL_SIZE, min(width, (c + 1) * GRID_CELL_SIZE)
        if grid_state[r, c] == 0:
            gui_matrix[y1:y2, x1:x2] = [255, 180, 100]   # Ruta planificada (Celeste)

    for r in range(grid_rows + 1):
        y = min(height - 1, r * GRID_CELL_SIZE)
        gui_matrix[y, :, :] = [225, 225, 225]
    for c in range(grid_cols + 1):
        x = min(width - 1, c * GRID_CELL_SIZE)
        gui_matrix[:, x, :] = [225, 225, 225]

    ry1, ry2 = robot_r * GRID_CELL_SIZE, min(height, (robot_r + 1) * GRID_CELL_SIZE)
    rx1, rx2 = robot_c * GRID_CELL_SIZE, min(width, (robot_c + 1) * GRID_CELL_SIZE)
    gui_matrix[ry1:ry2, rx1:rx2] = [0, 0, 255]           # Robot (Rojo brillante)

    WebGUI.showNumpy(gui_matrix)

route_idx = 0
stuck_counter = 0

while True:
    Frequency.tick(50)

    pose = HAL.getPose3d()
    laser_data = HAL.getLaserData()
    if pose is None:
        continue

    rx, ry, ryaw = pose.x, pose.y, pose.yaw
    curr_r, curr_c = get_grid_pos(rx, ry)

    if grid_state[curr_r, curr_c] == 0:
        grid_state[curr_r, curr_c] = 2

    front_dist = 10.0
    if laser_data and hasattr(laser_data, 'values') and len(laser_data.values) > 0:
        centro = len(laser_data.values) // 2
        rango_front = laser_data.values[max(0, centro - 20):min(len(laser_data.values), centro + 20)]
        rango_valido = [v for v in rango_front if not math.isinf(v) and not math.isnan(v)]
        if rango_valido:
            front_dist = min(rango_valido)

    if route_idx >= len(planned_route):
        HAL.setV(0.0)
        HAL.setW(0.0)
        render_gui(curr_r, curr_c, route_idx)
        continue

    target_cell = planned_route[route_idx]
    target_px = target_cell[1] * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0
    target_py = target_cell[0] * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0
    target_x, target_y = map_to_gazebo(target_px, target_py)

    dx = target_x - rx
    dy = target_y - ry
    distance = math.hypot(dx, dy)
    target_yaw = math.atan2(dy, dx)
    yaw_error = normalize_angle(target_yaw - ryaw)

    if distance < 0.20 or (curr_r == target_cell[0] and curr_c == target_cell[1]):
        route_idx += 1
        stuck_counter = 0
    else:
        stuck_counter += 1
        if stuck_counter > 60:
            route_idx += 1
            stuck_counter = 0
            continue

        if front_dist < 0.15:
            HAL.setV(-0.05)
            HAL.setW(0.8)
        else:
            if abs(yaw_error) > 0.25:
                HAL.setV(0.0)
                HAL.setW(0.8 if yaw_error > 0 else -0.8)
            else:
                v_cmd = min(0.35, 0.6 * distance)
                w_cmd = 1.2 * yaw_error
                HAL.setV(v_cmd)
                HAL.setW(w_cmd)

    render_gui(curr_r, curr_c, route_idx)