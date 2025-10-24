import os
import random
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, Point
import pandas as pd
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping

# ── 0) FIXED SEED ─────────────────────────────────────────────
SEED = 20
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# === Parameters ===
N = 4
x_goal = np.array([1.5, 0.0], dtype=np.float64)
formation_offsets = {
    0: np.array([ 0.2,  0.2]),
    1: np.array([ 0.2, -0.2]),
    2: np.array([-0.2, -0.2]),
    3: np.array([-0.2,  0.2]),
}
formation_edges = [(i,j) for i in formation_offsets for j in formation_offsets if i<j]
desired_d = { (i,j): np.linalg.norm(formation_offsets[i]-formation_offsets[j]) for (i,j) in formation_edges }
eps_max = { (i,j): 0.1 for (i,j) in formation_edges }

# Static obstacles
base_bean = np.array([(-0.2,0.1),(-0.1,0.2),(0.1,0.25),(0.3,0.2),
                      (0.35,0.0),(0.3,-0.2),(0.1,-0.25),(-0.1,-0.2),(-0.25,-0.05)])
translations = [np.array([0,0]), np.array([-0.45,-0.45]), np.array([-0.45,0.45])]
static_polys = []
for t in translations:
    pts = (base_bean + t).tolist()
    static_polys.append(Polygon(pts))
# Approximate polygons by circles
static_circles = []
for poly in static_polys:
    coords = np.array(poly.exterior.coords)
    center = coords.mean(axis=0)
    rad = 0.7*np.max(np.linalg.norm(coords - center, axis=1))
    static_circles.append((center, rad))

# Safety margin and timestep
dt = 0.1
safety_margin = 0.1

# CBF & control gains
kp = 1.0
k_eps = 10.0
w_eps = 10.0
beta1 = lambda h: 20*h
beta2 = lambda h: 20*h
gamma_obs = 1000

# Neighbor sets and elastic tolerances
neighbors = {i: [] for i in range(N)}
eps = np.zeros((N,N), dtype=float)
for (i,j) in formation_edges:
    neighbors[i].append(j)
    neighbors[j].append(i)

# History recording
center_distances = []
formation_error  = []
obs_dist_min     = []   # will store list-of-lists: each entry = [np.float64(...) for each robot]
inter_robot_dist = []

# === Robotarium setup ===
initial_positions = np.array([[-1, -0.7, -0.7, -1], [0.1,0.1,-0.1,-0.1], [0,0,0,0]], dtype=float)
r = robotarium.Robotarium(
    number_of_robots=N,
    initial_conditions=initial_positions,
    show_figure=True,
    sim_in_real_time=True
)
si_to_uni,_ = create_si_to_uni_mapping()

# Plot obstacles as polygons and circles
for poly in static_polys:
    r.axes.add_patch(MplPolygon(poly.exterior.coords,
                                 facecolor='lightgray', edgecolor='black', alpha=0.5))
for (center, rad) in static_circles:
    r.axes.add_patch(MplCircle(center, rad + safety_margin,
                               fill=False, edgecolor='red', linestyle='--'))
# Plot target
r.axes.plot(x_goal[0], x_goal[1], marker='o', color='red', markersize=15)
r.axes.text(x_goal[0]-0.05, x_goal[1]+0.1, 'Target', color='red', fontsize=12, ha='center')
# Formation outline
form_line = Line2D([], [], linestyle='--', color='blue')
r.axes.add_line(form_line)

# === Main loop ===
for step in range(1000):
    poses = r.get_poses()
    x = poses[0:2,:]

    # Compute min distance to obstacle circles (one np.float64 per robot)
    dmins_per_robot = []
    for i in range(N):
        # distances to each static circle for robot i
        d_i = [np.linalg.norm(x[:,i] - c) - (rad + safety_margin)
               for (c, rad) in static_circles]
        # min distance from robot i to any static circle as np.float64
        if len(d_i) > 0:
            dmins_per_robot.append(np.float64(np.min(np.array(d_i))))
        else:
            dmins_per_robot.append(np.float64(np.inf))
    obs_dist_min.append(dmins_per_robot)

    U_exec = np.zeros((2,N))
    # Solve per-robot QP-CBF
    for i in range(N):
        # Nominal control to formation at goal
        p_des = x_goal + formation_offsets[i]
        u_nom = kp * (p_des - x[:,i])
        v_nom = np.array([-k_eps*eps[i,j] for j in neighbors[i]])

        # Decision variables
        u_i = cp.Variable(2)
        v_i = cp.Variable(len(neighbors[i]))
        cost = cp.sum_squares(u_i - u_nom) + w_eps*cp.sum_squares(v_i - v_nom)
        cons = []
        # Obstacle CBFs using circle approx
        for (c, rad) in static_circles:
            dvec = x[:,i] - c
            h_o = np.linalg.norm(dvec)**2 - (rad + safety_margin)**2
            grad = dvec / (np.linalg.norm(dvec)+1e-6)
            cons.append(grad @ u_i >= -gamma_obs * h_o)
        # Formation CBFs + elastic edges
        for idx,j in enumerate(neighbors[i]):
            diff = x[:,i] - x[:,j]
            norm2 = float(diff.dot(diff))
            ε_ij, ε_ji = eps[i,j], eps[j,i]
            d_up = desired_d[(min(i,j),max(i,j))] + ε_ij + ε_ji
            d_low= desired_d[(min(i,j),max(i,j))] - ε_ij - ε_ji
            # upper bound CBF
            h_u = -norm2 + d_up**2
            L_u = -2*diff @ u_i + 2*d_up * v_i[idx]
            cons.append(L_u >= -beta1(h_u))
            # lower bound CBF
            h_l = norm2 - d_low**2
            L_l =  2*diff @ u_i + 2*d_low * v_i[idx]
            cons.append(L_l >= -beta1(h_l))
            # tolerance limits
            h_eu = eps_max[(min(i,j),max(i,j))] - (ε_ij + ε_ji)
            cons.append(-v_i[idx] >= -beta2(h_eu))
            cons.append(v_i[idx] >= -beta2(ε_ij))
        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True)

        # Extract solution
        U_exec[:,i] = u_i.value if u_i.value is not None else u_nom
        v_exec = v_i.value if v_i.value is not None else v_nom
        # Update tolerances
        for idx,j in enumerate(neighbors[i]):
            eps[i,j] += v_exec[idx] * dt
            eps[j,i] = eps[i,j]

    # Send commands to Robotarium
    dxu = si_to_uni(U_exec, poses)
    r.set_velocities(range(N), dxu)
    r.step()

    # Logging
    center = x.mean(axis=1)
    center_distances.append(np.linalg.norm(center - x_goal))
    errs = [np.linalg.norm((x[:,a]-x[:,b]) - (formation_offsets[a]-formation_offsets[b]))
            for (a,b) in formation_edges]
    formation_error.append(max(errs))
    inter_robot_dist.append([np.linalg.norm(x[:,i]-x[:,j]) for i in range(N) for j in range(i+1,N)])
    # Update formation outline
    xs = [x[0,i] for i in range(N)] + [x[0,0]]
    ys = [x[1,i] for i in range(N)] + [x[1,0]]
    form_line.set_data(xs, ys)

r.call_at_scripts_end()
