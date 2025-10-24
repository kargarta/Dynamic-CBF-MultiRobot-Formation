import os, random
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from random import sample
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, Point
from shapely import affinity
import matplotlib.pyplot as plt
import pandas as pd



# ── 0) FIXED SEED ─────────────────────────────────────────────
SEED = 20
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping

# === Hyperparams ===
N              = 4

d_safe         = 0.1

d_obs_safe     = 0.08

d_form_margin  = 0.1

gamma          = 43

gamma_f_base   = 30    # base formation CBF gain

gamma_f_rec    = 100   # recovery formation CBF gain

epsilon_clf    = 15.0

vmax           = 2.75

d_gate         = 0.3

# cost weights
p_clf          = 200.0
p_form_base    = 1.0    # base formation cost weight
p_form_rec     = 25.0   # recovery formation cost weight
p_obs          = 90.0
p_clf2         = 30.0

# analytic recovery gain for U_nom
K_recovery     = 0.2

# replay buffer / RL params
BUFFER_SIZE    = 50000
BATCH_SIZE     = 128
ACTOR_LR       = 1e-4
CRITIC_LR      = 1e-3
GAMMA          = 0.99
POLYAK         = 0.995
UPDATE_EVERY   = 5

# === Replay buffer ===
Transition = namedtuple('Transition', ('s','a','r','s2'))
class ReplayBuffer:
    def __init__(self, maxlen): self.buf = deque(maxlen=maxlen)
    def push(self, *args):      self.buf.append(Transition(*args))
    def sample(self, bs):       return sample(self.buf, bs)
    def __len__(self):         return len(self.buf)
buffer = ReplayBuffer(BUFFER_SIZE)

# === Actor & Critic ===
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim,256), nn.ReLU(),
            nn.Linear(256,256),   nn.ReLU(),
            nn.Linear(256,a_dim), nn.Tanh()
        )
    def forward(self, s): return self.net(s)

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim+a_dim,256), nn.ReLU(),
            nn.Linear(256,256),          nn.ReLU(),
            nn.Linear(256,1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s,a],dim=1))

# Augmented state dimension: positions (2N) + goal (2) + obstacle info (4N)
state_dim  = 2*N + 2 + 4*N
action_dim = 2*N

actor       = Actor(state_dim, action_dim)
critic      = Critic(state_dim, action_dim)
actor_targ  = Actor(state_dim, action_dim)
critic_targ = Critic(state_dim, action_dim)
actor_targ.load_state_dict(actor.state_dict())
critic_targ.load_state_dict(critic.state_dict())

opt_actor  = optim.Adam(actor.parameters(),  lr=ACTOR_LR)
opt_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)
mse_loss   = nn.MSELoss()

# === Formation & Env ===
formation_offsets = {
    0: np.array([ 0.2,  0.2]),
    1: np.array([ 0.2, -0.2]),
    2: np.array([-0.2, -0.2]),
    3: np.array([-0.2,  0.2]),
}
formation_pairs = [
    (i, j, formation_offsets[i] - formation_offsets[j])
    for i in formation_offsets for j in formation_offsets if i<j
]

# base static obstacles (bean-shaped)
base_bean = np.array([
    (-0.2,0.1),(-0.1,0.2),(0.1,0.25),(0.3,0.2),
    (0.35,0.0),(0.3,-0.2),(0.1,-0.25),(-0.1,-0.2),(-0.25,-0.05)
])
base_bean[:, 0] -= 0.1
translations = [np.array([0,0]), np.array([-0.45,-0.45]), np.array([-0.45,0.45])]
static_polys = []
for t in translations:
    scaled = (base_bean - base_bean.mean(axis=0))*0.9 + base_bean.mean(axis=0)
    pts = (scaled + t).tolist()
    static_polys.append(Polygon(pts))

# dynamic obstacles: simple linear moving rectangles
dynamic_polys = []
# define two dynamic obstacles with velocity
for center, size, vel in [([1.4, 0.6], [0.2,0.1], [-0.043, -0.00]), ([ 0.7,0.7], [0.2,0.1], [-0.000, -0.060])]:
    cx, cy = center; w, h = size; vx, vy = vel
    pts = [[cx-w, cy-h],[cx+w,cy-h],[cx+w,cy+h],[cx-w,cy+h]]
    dynamic_polys.append({
        'poly': Polygon(pts),
        'vel': np.array([vx,vy])
    })

x_goal = np.array([1.5, 0.0], dtype=np.float32)
initial_positions = np.array([
    [-1, -0.7, -0.7, -1],
    [ 0.1,  0.1, -0.1, -0.1],
    [ 0.0,  0.0,  0.0,  0.0]
], dtype=np.float32)

# Helper: extract obstacle info into vector of length 4N
def get_obs_info(x, static_polys, dynamic_polys):
    info = []
    # static as before
    for i in range(N):
        pt = Point(x[0,i], x[1,i])
        best = None
        for poly in static_polys:
            sdf = pt.distance(poly)
            inside = poly.contains(pt)
            signed = -sdf if inside else sdf
            if best is None or abs(signed)<abs(best[0]):
                nearest = np.array(poly.exterior.interpolate(
                    poly.exterior.project(pt)).coords[0])
                vec = nearest - x[:,i]
                norm = np.linalg.norm(vec)+1e-6
                unit = vec/norm
                best = (signed, unit[0], unit[1], 1.0 if inside else 0.0)
        # dynamic obstacles
        for dob in dynamic_polys:
            poly = dob['poly']; sdf = pt.distance(poly)
            inside = poly.contains(pt); signed_dyn = -sdf if inside else sdf
            if best is None or abs(signed_dyn)<abs(best[0]):
                nearest = np.array(poly.exterior.interpolate(
                    poly.exterior.project(pt)).coords[0])
                vec = nearest - x[:,i]
                norm = np.linalg.norm(vec)+1e-6
                unit = vec/norm
                best = (signed_dyn, unit[0], unit[1], 1.0 if inside else 0.0)
        info.extend(best)
    return np.array(info, dtype=np.float32)

# state flags
passed_flag      = False
slack_mode       = False
recovery_mode    = False
pass_timer       = 0
pass_clear_steps = 1
front_idxs       = {0,1}
back_idxs        = set(range(N)) - front_idxs
slack_edges      = {(0,1),(3,2)}

# Robotarium setup
r = robotarium.Robotarium(
    number_of_robots   = N,
    initial_conditions = initial_positions,
    show_figure        = True,
    sim_in_real_time   = True
)
si_to_uni, _ = create_si_to_uni_mapping()
# draw static obstacles & goal
for poly in static_polys:
    r.axes.add_patch(MplPolygon(poly.exterior.coords, facecolor='lightgray', edgecolor='black', alpha=0.5))

r.axes.plot(x_goal[0], x_goal[1], marker='o', color='red', markersize=20)

r.axes.text(
    x_goal[0]-0.05, x_goal[1] + 0.1,
    "Target",
    color='red',
    fontsize=12,
    fontweight='bold',
    ha='center',
    va='bottom',
    rotation=0
)
# dynamic obstacles patches
dyn_patches = []
for dob in dynamic_polys:
    patch = MplPolygon(dob['poly'].exterior.coords, facecolor='pink', edgecolor='black', alpha=0.6)
    r.axes.add_patch(patch)
    dyn_patches.append(patch)
# formation outline
form_line = Line2D([], [], linestyle='--', color='blue')
r.axes.add_line(form_line)

prev_s = prev_a = prev_r = None
step = 0
center_distances = []
d_obs_history = []                # to store min distance to obstacles per step
inter_robot_history = []          # to store inter-robot distances per step
steps =1000

U_exec_history = []   # will collect arrays of shape (2, N)
U_exec_history       = []
recovery_history     = []
slack_history        = []
passed_history       = []
formation_error_hist = []

dt = 0.1  # time step for dynamic update
for _ in range(1000):
    # update dynamic obstacles
    for dob, patch in zip(dynamic_polys, dyn_patches):
        dob['poly'] = affinity.translate(dob['poly'], xoff=dob['vel'][0]*dt, yoff=dob['vel'][1]*dt)
        patch.set_xy(dob['poly'].exterior.coords)

    poses = r.get_poses()
    x = poses[0:2,:]
    # augmented state: positions, goal, obstacle info
    obs_info = get_obs_info(x, static_polys, dynamic_polys)
    s = np.concatenate([x.flatten(), x_goal, obs_info]).astype(np.float32)

    # store experience
    if prev_s is not None:
        buffer.push(prev_s, prev_a, prev_r, s)

    # obstacle distances (for flags)
    d_obs = np.array([min([
        Point(x[0,i],x[1,i]).distance(poly)*( -1 if poly.contains(Point(x[0,i],x[1,i])) else 1)
        for poly in static_polys + [dob['poly'] for dob in dynamic_polys]
    ]) for i in range(N)])

    

    # flag logic unchanged...
    if not passed_flag:
        if all(d_obs > d_obs_safe + d_gate):
            pass_timer += 1
            if pass_timer >= pass_clear_steps:
                passed_flag   = True
                recovery_mode = True
        else:
            pass_timer = 0

    front_clear = any(d_obs[i] > d_obs_safe + d_gate for i in front_idxs)
    if front_clear:
        slack_mode    = False
        recovery_mode = True
    else:
        slack_mode = True

    # nominal control selection
    if recovery_mode:
        U_nom = np.zeros((2,N))
        for i in range(N):
            p_des = x_goal + formation_offsets[i]
            U_nom[:,i] = -K_recovery * (x[:,i] - p_des)
    else:
        with torch.no_grad(): u_rl = actor(torch.from_numpy(s).unsqueeze(0))
        U_nom = u_rl.cpu().numpy().reshape(2,N) * vmax

    # solve per-robot QPs
    U_exec = np.zeros((2,N))

    for i in range(N):
        u_i = cp.Variable(2); δ_i = cp.Variable(nonneg=True)
        s_i = cp.Variable(nonneg=True); o_i = cp.Variable(nonneg=True)
        κ_i = cp.Variable(nonneg=True)

        p_form = p_form_rec if recovery_mode else p_form_base
        cost = (cp.sum_squares(u_i - U_nom[:,i]) + p_clf*δ_i + p_form*s_i + p_obs*o_i + p_clf2*κ_i)
        cons = []
 
        if recovery_mode and slack_mode:
        # inter-robot safety
            for j in range(N):
                if i==j: continue
                d = x[:,i]-x[:,j]; h = d.dot(d)-d_safe**2
                cons.append((-2*d)@(u_i-U_nom[:,j]) <= gamma*h )

        # obstacle CBF including time-derivative for dynamic obstacles
        if not recovery_mode:
            for poly, is_dynamic, v_obs in [ (poly, False, np.zeros(2)) for poly in static_polys ] + [ (dob['poly'], True, dob['vel']) for dob in dynamic_polys ]:
                pt = Point(*x[:,i])
                sdf = pt.distance(poly); inside = poly.contains(pt)
                h = sdf - d_obs_safe
                grad = np.zeros(2)
                if sdf>1e-4:
                    nearest = np.array(poly.exterior.interpolate(poly.exterior.project(pt)).coords[0])
                    grad = (x[:,i] - nearest)/np.linalg.norm(x[:,i] - nearest)
                # time-derivative term
                dh_dt = -grad.dot(v_obs) if is_dynamic else 0.0
                cons.append(grad @ u_i + 150*dh_dt >= -gamma*(h) - 1*o_i)

        # formation & goal CBFs unchanged...
        gamma_f = gamma_f_rec if recovery_mode else gamma_f_base
        for (a,b,dab) in formation_pairs:
            if i not in (a,b): continue
            if slack_mode and (a,b) in slack_edges: continue
            diff = (x[:,a]-x[:,b]) - dab; h_f = d_form_margin**2 - diff.dot(diff)
            sign = 1 if i==a else -1
            if recovery_mode:
                cons.append(sign*(2*diff)@u_i >= -gamma_f*h_f)
            else:
                cons.append(sign*(2*diff)@u_i + s_i >= -gamma_f*h_f)

        a_i = x[:,i]-(x_goal+formation_offsets[i]); V_i=0.5*a_i.dot(a_i)
        cons.append(a_i@u_i <= -epsilon_clf*V_i + δ_i + κ_i)
        cons.append(cp.norm(u_i, 'inf') <= vmax)

        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True)
        U_exec[:,i] = u_i.value if u_i.value is not None else U_nom[:,i]
    U_exec_history.append(U_exec.copy())
    recovery_history.append(int(recovery_mode))
    slack_history.append(int(slack_mode))
    passed_history.append(int(passed_flag))
    errs = []
    for (a, b, dab) in formation_pairs:
        diff = (x[:, a] - x[:, b]) - dab
        errs.append(np.linalg.norm(diff))
    formation_error_hist.append(max(errs))  # or np.mean(errs)
    dxu = si_to_uni(U_exec, poses)
    r.set_velocities(range(N), dxu)
    r.step()

    # RL updates as before...
    center = x.mean(axis=1)
    center_distances.append(np.linalg.norm(center - x_goal))
    r_t = -np.linalg.norm(center - x_goal)
    if prev_s is not None:
        buffer.push(prev_s, prev_a, prev_r, s)
    prev_s, prev_a, prev_r = s, U_nom.flatten(), r_t

    if step % UPDATE_EVERY == 0 and len(buffer) >= BATCH_SIZE:
        S  = torch.tensor([b.s for b in buffer.sample(BATCH_SIZE)], dtype=torch.float32)
        A  = torch.tensor([b.a for b in buffer.sample(BATCH_SIZE)], dtype=torch.float32)
        R  = torch.tensor([b.r for b in buffer.sample(BATCH_SIZE)], dtype=torch.float32).unsqueeze(1)
        S2 = torch.tensor([b.s2 for b in buffer.sample(BATCH_SIZE)], dtype=torch.float32)
        with torch.no_grad():
            A2 = actor_targ(S2); Q2 = critic_targ(S2, A2)
            y  = R + GAMMA * Q2
        loss_c = mse_loss(critic(S, A), y)
        opt_critic.zero_grad(); loss_c.backward(); opt_critic.step()
        loss_a = -critic(S, actor(S)).mean()
        opt_actor.zero_grad(); loss_a.backward(); opt_actor.step()
        for p, pt in zip(actor.parameters(), actor_targ.parameters()): pt.data.mul_(POLYAK).add_(p.data, alpha=1-POLYAK)
        for p, pt in zip(critic.parameters(), critic_targ.parameters()): pt.data.mul_(POLYAK).add_(p.data, alpha=1-POLYAK)

    if recovery_mode and all(np.linalg.norm((x[:,a] - x[:,b]) - dab) < 0.01 for (a,b,dab) in formation_pairs) and np.linalg.norm(center - x_goal) < 0.1:
        recovery_mode = False

    form_line.set_color('gray' if slack_mode else ('blue' if recovery_mode else 'red'))
    xs = [x[0,i] for i in [0,1,2,3,0]]
    ys = [x[1,i] for i in [0,1,2,3,0]]
    form_line.set_data(xs, ys)

    d_obs = np.array([min([
        Point(x[0,i],x[1,i]).distance(poly)*(-1 if poly.contains(Point(x[0,i],x[1,i])) else 1)
        for poly in static_polys + [dob['poly'] for dob in dynamic_polys]
    ]) for i in range(N)])

    # Log distances
    d_obs_history.append(d_obs.copy())

    # Compute and log inter-robot distances
    d_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            d_pairs.append(np.linalg.norm(x[:,i] - x[:,j]))
    inter_robot_history.append(d_pairs)


    step += 1

r.call_at_scripts_end()

U_exec_arr = np.array(U_exec_history)   # shape should be (1000, 2, N)
recovery_arr       = np.array(recovery_history)     # (steps,)
slack_arr          = np.array(slack_history)        # (steps,)
passed_arr         = np.array(passed_history)       # (steps,)
formation_err_arr  = np.array(formation_error_hist) # (steps,)


plt.figure()
plt.plot(center_distances)
plt.xlabel('Time step')
plt.ylabel('Center Distance to Goal')
plt.title('Formation Center Convergence')
plt.grid(True)

d_obs_arr = np.array(d_obs_history)              # shape (steps, N)
inter_arr = np.array(inter_robot_history)        # shape (steps, n_pairs)

# Plot 1: minimum distance to obstacles per robot
plt.figure()
for i in range(N):
    plt.plot(np.arange(steps), d_obs_arr[:, i], label=f'Robot {i}')
plt.axhline(y=d_obs_safe, color='k', linestyle='--', label='Safety Threshold')
plt.xlabel('Step')
plt.ylabel('Signed Distance to Nearest Obstacle')
plt.title('Min Distance to Obstacles (per Robot)')
plt.legend()
plt.grid(True)

# Plot 2: inter-robot distances for each pair
plt.figure()
pair_labels = [f'{i}-{j}' for i in range(N) for j in range(i+1, N)]
for idx, label in enumerate(pair_labels):
    plt.plot(np.arange(steps), inter_arr[:, idx], label=f'Pair {label}')
plt.xlabel('Step')
plt.ylabel('Distance Between Robots')
plt.title('Inter-Robot Distances')
plt.legend()
plt.grid(True)


plt.figure()
for i in range(N):
    mag = np.max(np.abs(U_exec_arr[:, :, i]), axis=1)
    plt.plot(mag, label=f'Robot {i}')
plt.xlabel('Step')
plt.ylabel('Max |uᵢ|')
plt.title('Control Command Magnitudes')
plt.legend()
plt.grid(True)


plt.figure()
plt.step(range(steps), recovery_arr, where='post', label='recovery_mode')
plt.step(range(steps), slack_arr,    where='post', label='slack_mode')
plt.step(range(steps), passed_arr,   where='post', label='passed_flag')
plt.ylim(-0.1, 1.1)
plt.xlabel('Step'); plt.ylabel('Flag (0 or 1)')
plt.title('Mode Flags Timeline')
plt.legend(); plt.grid(True)

# ── Plot 3: Formation‐error norm ────────────────────────────────
plt.figure()
plt.plot(formation_err_arr)
plt.xlabel('Step'); plt.ylabel('Max Pairwise Formation Error')
plt.title('Formation Error Norm Over Time')
plt.grid(True)


plt.show()

num_steps = len(center_distances)

# build dict of columns
data = {
    'step': np.arange(num_steps),
    'center_distance':          center_distances,
    'formation_error':          formation_err_arr,
    'recovery_mode':            recovery_arr,
    'slack_mode':               slack_arr,
    'passed_flag':              passed_arr
}

# per-robot obstacle distances
for i in range(d_obs_arr.shape[1]):
    data[f'd_obs_robot_{i}'] = d_obs_arr[:, i]

# inter-robot distances for each pair
pair_labels = [f'{i}-{j}' for i in range(N) for j in range(i+1, N)]
for idx, label in enumerate(pair_labels):
    data[f'inter_dist_{label}'] = inter_arr[:, idx]

# max control magnitude per robot
for i in range(U_exec_arr.shape[2]):
    max_u = np.max(np.abs(U_exec_arr[:, :, i]), axis=1)
    data[f'max_u_robot_{i}'] = max_u

# create and save DataFrame
df = pd.DataFrame(data)
csv_path = os.path.join(os.getcwd(), 'Ours_dynamic.csv')
df.to_csv(csv_path, index=False)
print(f"Saved all plotting data to {csv_path}")

input("Press Enter to close...")  # Pause until user input