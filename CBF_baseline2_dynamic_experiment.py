import os
import random
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle
from shapely.geometry import Polygon, Point
from shapely import affinity
import pandas as pd

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping

# ── Reproducibility ─────────────────────────────────────────
SEED = 20
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── Parameters ──────────────────────────────────────────────
N = 4
k_nom       = 1.5       # nominal SI gain
alpha_obs   = 50.0      # CBF gain for obstacles
alpha_form  = 30.0      # CBF gain for formation
d_safe      = 0.1       # inter-robot min distance
circle_scale = 0.9      # shrink factor for circle approximation

time_horizon = 1200     # simulation steps
safety_margin = 0.0     # if you want extra margin on plotted circles, increase

# formation offsets (for shape around leader)
formation_offsets = {
    0: np.array([ 0.3,  0.3]),
    1: np.array([ 0.3, -0.3]),
    2: np.array([-0.3, -0.3]),
    3: np.array([-0.3,  0.3]),
}
formation_pairs = [
    (i, j, formation_offsets[i] - formation_offsets[j])
    for i in formation_offsets for j in formation_offsets if i < j
]


# static obstacles as polygons
base_bean = 1.2*np.array([
    (-0.2,0.15),(-0.15,0.18),(0.1,0.25),(0.3,0.2),
    (0.35,0.0),(0.3,-0.2),(0.1,-0.25),(-0.1,-0.2),(-0.25,-0.05)
])
base_bean[:, 0] -= 0.3
translations = [np.array([0,0]), np.array([-0.6,-0.6]), np.array([-0.6,0.6])]
static_polys = []
for t in translations:
    scaled = (base_bean - base_bean.mean(axis=0))*0.9 + base_bean.mean(axis=0)
    pts = (scaled + t).tolist()
    static_polys.append(Polygon(pts))

# approximate each polygon by minimal enclosing circle (scaled)
circle_obstacles = []
for poly in static_polys:
    coords = np.array(poly.exterior.coords)
    center = coords.mean(axis=0)
    raw_radius = np.max(np.linalg.norm(coords - center, axis=1))
    radius = raw_radius * circle_scale
    circle_obstacles.append((center, radius))

# --- DYNAMIC obstacles (added) ---
# define moving rectangular obstacles (polygon + velocity)
dynamic_polys = []
dyn_specs = [([ 0.6,-0.7], [0.2,0.1], [-0.000, 0.040]), ([ 0.7,0.7], [0.2,0.1], [-0.000, -0.035])]

for center, size, vel in dyn_specs:
    cx, cy = center
    w, h = size
    vx, vy = vel
    pts = [[cx-w, cy-h], [cx+w, cy-h], [cx+w, cy+h], [cx-w, cy+h]]
    dynamic_polys.append({
        'poly': Polygon(pts),
        'vel': np.array([vx, vy], dtype=float)
    })

# Approximate dynamic polygons by circles (dicts so we can update centers)
dynamic_circles = []
for dob in dynamic_polys:
    coords = np.array(dob['poly'].exterior.coords)
    center = coords.mean(axis=0)
    rad = 1 * np.max(np.linalg.norm(coords - center, axis=1)) * circle_scale
    dynamic_circles.append({'center': center.copy(), 'rad': float(rad), 'vel': dob['vel'].copy()})

# goal/target
target = np.array([1.3, 0.0], dtype=np.float64)

# Robotarium init
initial_pos = np.vstack([
    np.array([-1.1, -0.6, -0.6, -1.1])[:N],
    np.array([ 0.25,  0.25, -0.25, -0.25])[:N],
    np.zeros(N)
]).astype(np.float64)

r = robotarium.Robotarium(
    number_of_robots   = N,
    initial_conditions = initial_pos,
    show_figure        = True,
    sim_in_real_time   = True
)
si_to_uni, _ = create_si_to_uni_mapping()

# visualize static obstacles and target
for poly, (center, radius) in zip(static_polys, circle_obstacles):
    r.axes.add_patch(
        MplPolygon(poly.exterior.coords,
                   facecolor='lightgray',
                   edgecolor='black',
                   alpha=0.5)
    )
    r.axes.add_patch(
        MplCircle(center,
                  radius + safety_margin,
                  fill=False,
                  edgecolor='red',
                  linestyle='--')
    )

# visualize dynamic polygons & their circle approximations (keep patches for updates)
dyn_poly_patches = []
dyn_circle_patches = []
for dob, dc in zip(dynamic_polys, dynamic_circles):
    patch = MplPolygon(dob['poly'].exterior.coords, facecolor='pink', edgecolor='black', alpha=0.6)
    r.axes.add_patch(patch)
    dyn_poly_patches.append(patch)
    circ = MplCircle(tuple(dc['center']), dc['rad'] + safety_margin, fill=False, edgecolor='orange', linestyle='--')
    r.axes.add_patch(circ)
    dyn_circle_patches.append(circ)

r.axes.plot(target[0],
            target[1],
            marker='o',
            color='red',
            markersize=20)
r.axes.text(
    target[0] - 0.05,
    target[1] + 0.1,
    "Target",
    color='red',
    fontsize=12,
    fontweight='bold',
    ha='center',
    va='bottom'
)

# ── LOG STORAGE ──────────────────────────────────────────────
center_dist       = []  # scalar per step
formation_err     = []  # scalar per step
min_obs_dist      = []  # list of N per step
inter_robot_dist  = []  # list of M = N*(N-1)/2 per step
control_mag_hist  = []  # list of N per step

# ── MAIN LOOP ───────────────────────────────────────────────
for t in range(time_horizon):
    # update dynamic obstacles (translate polygons and update circle centers & patches)
    for idx, dob in enumerate(dynamic_polys):
        vel = dob['vel']
        dob['poly'] = affinity.translate(dob['poly'], xoff=vel[0]* (1.0/10.0), yoff=vel[1]* (1.0/10.0))
        # Note: above uses a small step scale consistent with dt-like progression;
        # adjust factor if your Robotarium uses a different timestep.
        dyn_poly_patches[idx].set_xy(dob['poly'].exterior.coords)
        coords = np.array(dob['poly'].exterior.coords)
        center = coords.mean(axis=0)
        dynamic_circles[idx]['center'] = center.copy()
        # update displayed circle patch center
        dyn_circle_patches[idx].set_center(tuple(center))

    poses = r.get_poses()
    p_all = poses[0:2, :]  # shape (2, N)

    # Nominal SI control
    U_nom = np.zeros((2, N))
    U_nom[:, 0] = -k_nom * (p_all[:, 0] - target)  # leader
    for i in range(1, N):
        pos_des = p_all[:, 0] + formation_offsets[i]
        U_nom[:, i] = -k_nom * (p_all[:, i] - pos_des)

    # CBF-QP per robot
    U_exec = np.zeros_like(U_nom)
    obs_dists = []
    inter_dists = []
    cm_step = []

    for i in range(N):
        u = cp.Variable(2)
        cost = cp.sum_squares(u - U_nom[:, i])
        cons = []

        # obstacle CBFs + record distances (static circles)
        dists = []
        for center, radius in circle_obstacles:
            diff = p_all[:, i] - center
            h = diff.dot(diff) - (radius + safety_margin)**2
            grad = 2 * diff
            cons.append(grad @ u >= -alpha_obs * h)
            dists.append(np.linalg.norm(diff) - (radius + safety_margin))

        # include dynamic circle obstacles (use their updated centers)
        for dc in dynamic_circles:
            center = dc['center']
            radius = dc['rad']
            diff = p_all[:, i] - center
            h = diff.dot(diff) - (radius + safety_margin)**2
            grad = 2 * diff
            cons.append(grad @ u >= -alpha_obs * h)
            dists.append(np.linalg.norm(diff) - (radius + safety_margin))

        # record minimum distance to any obstacle for this robot
        if len(dists) > 0:
            obs_dists.append(min(dists))
        else:
            obs_dists.append(float('inf'))

        # inter-robot CBFs + record pairwise distances
        for j in range(N):
            if i == j:
                continue
            diff = p_all[:, i] - p_all[:, j]
            h_ij = diff.dot(diff) - d_safe**2
            # the original used (u - U_nom[:, j]) in the linear term; we follow that pattern
            cons.append(2 * diff @ (u - U_nom[:, j]) >= -alpha_form * h_ij)
            if j > i:
                inter_dists.append(np.linalg.norm(diff))

        # solve QP
        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True)
        U_i = u.value if u.value is not None else U_nom[:, i]
        U_exec[:, i] = U_i
        cm_step.append(np.linalg.norm(U_i))

    # apply velocities
    dxu = si_to_uni(U_exec, poses)
    r.set_velocities(range(N), 0.12*dxu)
    r.step()

    # log
    center_dist.append(np.linalg.norm(p_all.mean(axis=1) - target))
    formation_err.append(
        max(np.linalg.norm((p_all[:, a] - p_all[:, b]) - dab)
            for (a, b, dab) in formation_pairs)
    )
    min_obs_dist.append(obs_dists)
    inter_robot_dist.append(inter_dists)
    control_mag_hist.append(cm_step)

r.call_at_scripts_end()

