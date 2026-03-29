import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
from scipy.optimize import least_squares
import sympy as sp

def rotx(a):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(a), -sp.sin(a)],
        [0, sp.sin(a),  sp.cos(a)]
    ])

def roty(a):
    return sp.Matrix([
        [ sp.cos(a), 0, sp.sin(a)],
        [0, 1, 0],
        [-sp.sin(a), 0, sp.cos(a)]
    ])

def rotz(a):
    return sp.Matrix([
        [sp.cos(a), -sp.sin(a), 0],
        [sp.sin(a),  sp.cos(a), 0],
        [0, 0, 1]
    ])

def H_rot_z(q):
    return sp.Matrix([
        [sp.cos(q), -sp.sin(q), 0, 0],
        [sp.sin(q),  sp.cos(q), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def H_xyzrpy(pos, rpy):
    R = rotz(rpy[2]) * roty(rpy[1]) * rotx(rpy[0])
    T = sp.eye(4)
    T[:3, :3] = R
    T[:3, 3] = sp.Matrix(pos)
    return T

def S(x):
    return sp.Rational(str(x))

q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5', real=True)
q_syms = [q1, q2, q3, q4, q5]
T_world_base  = H_xyzrpy([0, 0, 0], [0, 0, sp.pi])
T_base_shldr  = H_xyzrpy([0, S(-0.0452), S(0.0165)], [0, 0, 0]) * H_rot_z(q1)
T_shldr_upper = H_xyzrpy([0, S(-0.0306), S(0.1025)], [0, -sp.pi/2, 0]) * H_rot_z(q2)
T_upper_lower = H_xyzrpy([S(0.11257), S(-0.028), 0], [0, 0, 0]) * H_rot_z(q3)
T_lower_wrist = H_xyzrpy([S(0.0052), S(-0.1349), 0], [0, 0, sp.pi/2]) * H_rot_z(q4)
T_wrist_grip  = H_xyzrpy([S(-0.0601), 0, 0], [0, -sp.pi/2, 0]) * H_rot_z(q5)
T_grip_center = H_xyzrpy([0, 0, S(0.075)], [0, 0, 0])
T_fk = sp.trigsimp(
    T_world_base
    * T_base_shldr
    * T_shldr_upper
    * T_upper_lower
    * T_lower_wrist
    * T_wrist_grip
    * T_grip_center
)
R_fk = sp.trigsimp(T_fk[:3, :3])
p_fk = sp.trigsimp(T_fk[:3, 3])
fk_pos_func = sp.lambdify((q1, q2, q3, q4, q5), p_fk, "numpy")
fk_rot_func = sp.lambdify((q1, q2, q3, q4, q5), R_fk, "numpy")
z_tool_sym = R_fk[:, 2]
task_sym = sp.Matrix([
    p_fk[0],
    p_fk[1],
    p_fk[2],
    z_tool_sym[0],
    z_tool_sym[1]
])
task_func = sp.lambdify((q1, q2, q3, q4, q5), task_sym, "numpy")
def rotx_np(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa,  ca]
    ], dtype=float)
def roty_np(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ ca, 0, sa],
        [  0, 1,  0],
        [-sa, 0, ca]
    ], dtype=float)
def rotz_np(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, -sa, 0],
        [sa,  ca, 0],
        [ 0,   0, 1]
    ], dtype=float)
def rpy_to_R(roll, pitch, yaw):
    return rotz_np(yaw) @ roty_np(pitch) @ rotx_np(roll)
def pose_to_task(target_pose):
    x, y, z, roll, pitch, yaw = target_pose
    R_des = rpy_to_R(roll, pitch, yaw)
    z_des = R_des[:, 2]
    return np.array([x, y, z, z_des[0], z_des[1]], dtype=float)
def evaluate_solution(q):
    q = np.array(q, dtype=float)
    pos = np.array(fk_pos_func(*q), dtype=float).reshape(3)
    R = np.array(fk_rot_func(*q), dtype=float).reshape(3, 3)
    z_tool = R[:, 2]
    task = np.array(task_func(*q), dtype=float).reshape(5)
    return pos, R, z_tool, task
def inverse_kinematics_pose5(
    target_pose,
    initial_guess=None,
    bounds=None,
    pos_weight=1.0,
    ori_weight=0.4,
    pos_tol=5e-3,
    ori_tol=5e-2,
    max_nfev=300
):
    if initial_guess is None:
        initial_guess = np.zeros(5, dtype=float)
    else:
        initial_guess = np.array(initial_guess, dtype=float)
    if bounds is None:
        lower = -np.pi * np.ones(5)
        upper =  np.pi * np.ones(5)
    else:
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
    target_task = pose_to_task(target_pose)
    weights = np.array([
        pos_weight, pos_weight, pos_weight,
        ori_weight, ori_weight
    ], dtype=float)
    def residual(q):
        current_task = np.array(task_func(*q), dtype=float).reshape(5)
        err = current_task - target_task
        return weights * err
    result = least_squares(
        residual,
        x0=initial_guess,
        bounds=(lower, upper),
        method="trf",
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=max_nfev
    )
    q_sol = result.x
    pos, R, z_tool, task = evaluate_solution(q_sol)
    target_pos = np.array(target_pose[:3], dtype=float)
    target_R = rpy_to_R(target_pose[3], target_pose[4], target_pose[5])
    target_z = target_R[:, 2]
    pos_err_vec = pos - target_pos
    pos_err = np.linalg.norm(pos_err_vec)
    ori_err_vec = z_tool[:2] - target_z[:2]
    ori_err = np.linalg.norm(ori_err_vec)
    feasible = (pos_err < pos_tol) and (ori_err < ori_tol)
    info = {
        "q": q_sol,
        "success_flag": result.success,
        "message": result.message,
        "cost": result.cost,
        "nfev": result.nfev,
        "position": pos,
        "rotation": R,
        "z_tool": z_tool,
        "target_z": target_z,
        "pos_err_vec": pos_err_vec,
        "pos_err": pos_err,
        "ori_err_vec": ori_err_vec,
        "ori_err": ori_err,
        "feasible": feasible
    }
    return q_sol, info, result

# --- Joint limits (from pickup_traj) ---
JOINT_LIMITS_RAD = np.array([
    [np.deg2rad(-135.0), np.deg2rad(135.0)],
    [np.deg2rad(-120.0), np.deg2rad(120.0)],
    [np.deg2rad(-120.0), np.deg2rad(120.0)],
    [np.deg2rad(-100.0), np.deg2rad(100.0)],
    [np.deg2rad(-180.0), np.deg2rad(180.0)],
], dtype=float)

class TrajPublisher(Node):
    def __init__(self, traj, dt=0.04, joint_offset=None):
        super().__init__('traj_publisher')
        self.traj = traj
        self.dt = dt
        self.idx = 0
        self.pub = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self.timer = self.create_timer(self.dt, self.publish_next)
        self.startup = True  # True for first pass (includes home), then False for looping only waypoints
        self.loop_start = None
        # EE trace marker
        self._marker_pub = self.create_publisher(Marker, 'ee_trace', 10)
        self._trace_marker = Marker()
        self._trace_marker.header.frame_id = 'world'
        self._trace_marker.ns     = 'ee_trace'
        self._trace_marker.id     = 0
        self._trace_marker.type   = Marker.LINE_STRIP
        self._trace_marker.action = Marker.ADD
        self._trace_marker.scale.x = 0.003
        self._trace_marker.color.r = 0.0
        self._trace_marker.color.g = 0.8
        self._trace_marker.color.b = 1.0
        self._trace_marker.color.a = 1.0
        # Joint offset (for real robot calibration)
        if joint_offset is None:
            self.joint_offset = np.zeros(6)
        else:
            self.joint_offset = np.array(joint_offset, dtype=float)

    def publish_next(self):
        if self.startup:
            if self.idx < len(self.traj):
                q = self.traj[self.idx]
                self.idx += 1
            else:
                self.startup = False
                n_interp = 30 - 1  # [1:] in build_joint_traj_from_cartesian
                self.loop_start = n_interp
                self.idx = self.loop_start
                q = self.traj[self.idx]
                self.idx += 1
        else:
            if self.idx < len(self.traj):
                q = self.traj[self.idx]
                self.idx += 1
            else:
                self.idx = self.loop_start
                q = self.traj[self.idx]
                self.idx += 1
        msg = JointTrajectory()
        pt = JointTrajectoryPoint()
        # Apply joint offset before publishing
        q_cmd = np.array(q, dtype=float) - self.joint_offset
        pt.positions = list(q_cmd)
        pt.time_from_start = Duration(sec=0, nanosec=int(self.dt*1e9))
        msg.points = [pt]
        self.pub.publish(msg)

        # --- EE trace marker update ---
        # Use only first 5 joints for FK
        pos = np.array(fk_pos_func(*q[:5]), dtype=float).reshape(3)
        tp = Point()
        tp.x, tp.y, tp.z = float(pos[0]), float(pos[1]), float(pos[2])
        self._trace_marker.points.append(tp)
        if len(self._trace_marker.points) > 2000:
            self._trace_marker.points.pop(0)
        self._trace_marker.header.stamp = self.get_clock().now().to_msg()
        self._marker_pub.publish(self._trace_marker)

# --- Cartesian Trajectory Generation ---
def interpolate_cartesian_traj(waypoints, steps_per_segment=50):
    """
    waypoints: list of (x, y, z, rot1, rot2)
    Returns: list of poses interpolated between waypoints
    """
    poses = []
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end = np.array(waypoints[i+1])
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            poses.append(start + (end - start) * t)
    poses.append(np.array(waypoints[-1]))
    return poses

# --- Build Joint Trajectory from Cartesian Waypoints ---
def build_joint_traj_from_cartesian(home_q, cartesian_waypoints, steps_per_segment=50):
    poses = interpolate_cartesian_traj(cartesian_waypoints, steps_per_segment)
    traj = [home_q.copy()]
    q_seed = home_q.copy()
    # Main Cartesian trajectory
    for pose in poses:
        q = numerical_ik(pose, q_seed)
        traj.append(q.copy())
        q_seed = q  # Use last solution as next seed for smoothness

    # Interpolate from home to first point in joint space
    n_interp = 30
    q_first = traj[1]
    to_start = [home_q + (q_first - home_q) * t for t in np.linspace(0, 1, n_interp, endpoint=False)][1:]
    # Full trajectory: to_start + main (no return to home)
    full_traj = list(to_start) + traj[1:]
    return full_traj

# --- Numerical IK helper (5DOF, roll, pitch, and yaw free) ---
def numerical_ik(pose, q_seed):
    x, y, z, roll, pitch = pose
    # Only constrain x, y, z, and pitch (z_tool_y)
    target_pose = [x, y, z, 0.0, pitch, 0.0]
    q_guess = q_seed[:5] if len(q_seed) >= 5 else np.zeros(5)
    pos_weight = 1.0
    ori_weight = 0.05
    weights = np.array([pos_weight, pos_weight, pos_weight, 0.0, ori_weight], dtype=float)
    def residual(q):
        current_task = np.array(task_func(*q), dtype=float).reshape(5)
        # Only constrain z_tool_y to match sin(pitch)
        target_task = np.array([x, y, z, 0.0, np.sin(pitch)], dtype=float)
        err = current_task - target_task
        return weights * err
    lower = JOINT_LIMITS_RAD[:, 0]
    upper = JOINT_LIMITS_RAD[:, 1]
    result = least_squares(
        residual,
        x0=q_guess,
        bounds=(lower, upper),
        method="trf",
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=600
    )
    q_sol = result.x
    pos, R, z_tool, task = evaluate_solution(q_sol)
    pos_err = np.linalg.norm(pos - np.array([x, y, z], dtype=float))
    pitch_err = np.abs(z_tool[1] - np.sin(pitch))
    feasible = (pos_err < 5e-3) and (pitch_err < 5e-2)
    print(f"[IK DEBUG] pose={pose}, pos_err={pos_err:.5f}, pitch_err={pitch_err:.5f}, feasible={feasible}")
    if len(q_seed) == 6:
        q_full = np.zeros(6)
        q_full[:5] = q_sol
        q_full[5] = q_seed[5]
        return q_full
    return q_sol

def run_trajectory():
    rclpy.init()
    # Home position matches URDF/sim: 0, 105°, -70°, -60°, 90° (6th is gripper)
    q_home = np.array([0, np.deg2rad(105), np.deg2rad(-70), np.deg2rad(-60), np.deg2rad(90), 0.0])
    # Example Cartesian waypoints: (x, y, z, roll, pitch)
    cartesian_waypoints = [
        (0.1, 0.25, 0.15, 0.0, 1.57),
        (0.1, 0.35, 0.15, 0.0, 1.57),
        (-0.1, 0.35, 0.15, 0.0, 1.57),
        (-0.1, 0.25, 0.15, 0.0, 1.57),  # Loop back to start
        (0.1, 0.25, 0.15, 0.0, 1.57),  # Loop back to start
    ]
    dt = 0.04
    traj = build_joint_traj_from_cartesian(q_home, cartesian_waypoints, steps_per_segment=50)
    joint_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Change these values to calibrate real robot offsets
    node = TrajPublisher(traj, dt=dt, joint_offset=joint_offset)
    rclpy.spin(node)
    rclpy.shutdown()

def main():
    run_trajectory()

if __name__ == '__main__':
    main()
