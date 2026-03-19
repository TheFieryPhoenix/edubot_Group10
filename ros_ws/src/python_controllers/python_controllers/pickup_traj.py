try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
except ImportError:
    rclpy = None
    # dummy base class so file can be imported outside of ROS
    class Node:
        def __init__(self, *args, **kwargs):
            pass

import numpy as np
import sympy as sp

try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
except ImportError:
    # define dummy message classes for non-ROS import scenarios
    class JointTrajectory:
        def __init__(self):
            self.header = type('h', (), {'stamp': None})
            self.points = []
    class JointTrajectoryPoint:
        def __init__(self):
            self.velocities = []

try:
    from sensor_msgs.msg import JointState
except ImportError:
    class JointState:
        def __init__(self):
            self.name = []
            self.position = []

# ----------------------------
# Basic rotations (symbolic)
# ----------------------------
def rotx(a):
    return sp.Matrix([[1,0,0],[0,sp.cos(a),-sp.sin(a)],[0,sp.sin(a),sp.cos(a)]])

def roty(a):
    return sp.Matrix([[sp.cos(a),0,sp.sin(a)],[0,1,0],[-sp.sin(a),0,sp.cos(a)]])

def rotz(a):
    return sp.Matrix([[sp.cos(a),-sp.sin(a),0],[sp.sin(a),sp.cos(a),0],[0,0,1]])

def rot_z_4(q):
    return sp.Matrix([
        [sp.cos(q), -sp.sin(q), 0, 0],
        [sp.sin(q),  sp.cos(q), 0, 0],
        [0,          0,         1, 0],
        [0,          0,         0, 1]
    ])

def thin_tform(pos, rpy):
    R = rotz(rpy[2]) * roty(rpy[1]) * rotx(rpy[0])
    T = sp.eye(4)
    T[:3,:3] = R
    T[:3,3] = sp.Matrix(pos)
    return T

# ----------------------------
# Build full symbolic transform T_world_gc(q)
# ----------------------------
def get_symbolic_T_world_gc():
    q1,q2,q3,q4,q5 = sp.symbols('q1 q2 q3 q4 q5', real=True)

    # Fixed transforms from assignment (same as your FK)
    T_world_base   = thin_tform([0,0,0], [0,0, sp.pi])  # assignment uses yaw=pi 
    T_base_sh      = thin_tform([0, -0.0452, 0.0165], [0,0,0]) * rot_z_4(q1)
    T_sh_ua        = thin_tform([0, -0.0306, 0.1025], [0, -1.57079, 0]) * rot_z_4(q2)
    T_ua_la        = thin_tform([0.11257, -0.028, 0], [0,0,0]) * rot_z_4(q3)
    T_la_wr        = thin_tform([0.0052, -0.1349, 0], [0,0, sp.pi/2]) * rot_z_4(q4)
    T_wr_gr        = thin_tform([-0.0601, 0, 0], [0, -sp.pi/2, 0]) * rot_z_4(q5)
    T_gr_gc        = thin_tform([0,0,0.075], [0,0,0])

    T = T_world_base * T_base_sh * T_sh_ua * T_ua_la * T_la_wr * T_wr_gr * T_gr_gc
    return T, (q1,q2,q3,q4,q5)

# ----------------------------
# Jacobian construction
# ----------------------------
def get_symbolic_jacobian():
    T, q = get_symbolic_T_world_gc()
    p = T[:3, 3]  # end-effector position

    # 1) Linear Jacobian: Jv = d p / d q  (Lecture 4: x=f(q) => δx = J δq) 
    Jv = p.jacobian(sp.Matrix(q))

    # 2) Rotational Jacobian:
    z_local = sp.Matrix([0,0,1])

    # compute intermediate transforms
    q1,q2,q3,q4,q5 = q

    T_world_base   = thin_tform([0,0,0], [0,0, sp.pi])
    T_base_sh_o    = thin_tform([0, -0.0452, 0.0165], [0,0,0])
    T_sh_ua_o      = thin_tform([0, -0.0306, 0.1025], [0, -1.57079, 0])
    T_ua_la_o      = thin_tform([0.11257, -0.028, 0], [0,0,0])
    T_la_wr_o      = thin_tform([0.0052, -0.1349, 0], [0,0, sp.pi/2])
    T_wr_gr_o      = thin_tform([-0.0601, 0, 0], [0, -sp.pi/2, 0])

    T0 = T_world_base
    T1 = T0 * (T_base_sh_o * rot_z_4(q1))
    T2 = T1 * (T_sh_ua_o   * rot_z_4(q2))
    T3 = T2 * (T_ua_la_o   * rot_z_4(q3))
    T4 = T3 * (T_la_wr_o   * rot_z_4(q4))
    T5 = T4 * (T_wr_gr_o   * rot_z_4(q5))

    R_world_j1 = (T0 * T_base_sh_o)[:3,:3]
    R_world_j2 = (T1 * T_sh_ua_o)[:3,:3]
    R_world_j3 = (T2 * T_ua_la_o)[:3,:3]
    R_world_j4 = (T3 * T_la_wr_o)[:3,:3]
    R_world_j5 = (T4 * T_wr_gr_o)[:3,:3]

    z1 = R_world_j1 * z_local
    z2 = R_world_j2 * z_local
    z3 = R_world_j3 * z_local
    z4 = R_world_j4 * z_local
    z5 = R_world_j5 * z_local

    Jw = sp.Matrix.hstack(z1,z2,z3,z4,z5)

    J = sp.Matrix.vstack(Jv, Jw)
    return J, Jv, Jw, q

# ----------------------------
# Numeric helpers
# ----------------------------

def jacobian_numeric_func():
    J, Jv, Jw, q = get_symbolic_jacobian()
    return sp.lambdify(q, J, modules="numpy")


def fk_position_numeric_func():
    T_sym, q_syms = get_symbolic_T_world_gc()
    p_sym = T_sym[:3, 3]
    return sp.lambdify(q_syms, p_sym, modules="numpy")


def jv_position_numeric_func():
    """Lambdified linear Jacobian Jv (3x5) – used only for startup IK solving."""
    T_sym, q_syms = get_symbolic_T_world_gc()
    p_sym = T_sym[:3, 3]
    Jv_sym = p_sym.jacobian(sp.Matrix(q_syms))
    return sp.lambdify(q_syms, Jv_sym, modules="numpy")


def ik_solve_position(p_target, q_init, fk_func, jv_func,
                      max_iter=200, tol=5e-4, lam=1e-2):
    """Iterative pseudoinverse IK: returns joint config that achieves p_target."""
    q = np.array(q_init, dtype=float)
    for _ in range(max_iter):
        p = np.array(fk_func(*q), dtype=float).reshape(3)
        err = p_target - p
        if np.linalg.norm(err) < tol:
            break
        Jv = np.array(jv_func(*q), dtype=float)
        Jv_pinv = Jv.T @ np.linalg.inv(Jv @ Jv.T + lam**2 * np.eye(3))
        q = clamp_q_to_limits(q + np.clip(Jv_pinv @ err, -0.3, 0.3))
    return q


def damped_pinv(A, lam=1e-2):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    return A.T @ np.linalg.inv(A @ A.T + (lam**2) * np.eye(m))


def cond_number(A, eps=1e-12):
    U, S, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    if S[-1] < eps:
        return np.inf
    return float(S[0] / S[-1])


def clamp_vec(v, v_max):
    return np.clip(v, -v_max, v_max)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_vec_to_pi(v):
    return wrap_to_pi(np.asarray(v, dtype=float))


# Per-joint position limits [min, max] in radians.
# These are used ONLY to gate outgoing velocity commands.
# Measured state is NEVER clamped – clamping raw sim output causes
# fake errors that spin joints in the wrong direction.
JOINT_LIMITS_RAD = np.array([
    [np.deg2rad(-135.0), np.deg2rad(135.0)],  # q1  shoulder rotation
    [np.deg2rad(-120.0), np.deg2rad(120.0)],  # q2  shoulder pitch
    [np.deg2rad(-120.0), np.deg2rad(120.0)],  # q3  elbow
    [np.deg2rad(-100.0), np.deg2rad(100.0)],  # q4  wrist pitch
    [np.deg2rad(-180.0), np.deg2rad(180.0)],  # q5  wrist roll
], dtype=float)


def clamp_q_to_limits(q):
    """Clamp joint positions to physical limits."""
    return np.clip(np.asarray(q, dtype=float),
                   JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


def gate_vel_at_limits(q, dq, margin=np.deg2rad(1.0)):
    """Zero out velocity components that would drive a joint past its limit."""
    q   = np.asarray(q,   dtype=float)
    dq  = np.asarray(dq,  dtype=float)
    out = dq.copy()
    lo  = JOINT_LIMITS_RAD[:, 0] + margin
    hi  = JOINT_LIMITS_RAD[:, 1] - margin
    out = np.where((q <= lo) & (out < 0.0), 0.0, out)
    out = np.where((q >= hi) & (out > 0.0), 0.0, out)
    return out


class ExampleTraj(Node):

    def __init__(self):
        super().__init__('example_trajectory')

        # joint home position – clamped to physical limits at construction
        self._HOME = clamp_q_to_limits(np.array([
            np.deg2rad(0), np.deg2rad(70),
            np.deg2rad(-40), np.deg2rad(-60),
            np.deg2rad(0)
        ], dtype=float))

        # state for velocity control
        self._q = self._HOME.copy()
        self._have_joint_state = False
        self._init_done = False
        self._init_tol = np.deg2rad(1.0)
        self._init_kp = 1.5
        self._post_home_settle_sec = 1.0
        self._home_done_time = None
        self._move_phase_timeout = 12.0
        self._sequence_started = False
        self._phase_index = 0
        self._phase_start_time = None
        self._last_phase_index_logged = -1
        self._phases = []
        self._sequence_complete = False
        self._repeat_sequence = True

        self._qdot_max       = 0.5     # nominal peak joint speed (rad/s) → sets move duration
        self._kp_joint       = 2.0     # joint-space tracking gain
        self._traj_q_start   = None    # joint config at start of current move segment
        self._traj_q_goal    = None    # joint config at end of current move segment
        self._phase_duration = None    # time for current segment [s]

        self._use_absolute_targets = True
        self._pickup_abs = np.array([0.3, 0.3, 0.01], dtype=float)
        self._dropoff_abs = np.array([0.0, 0.05, 0.0], dtype=float)
        self._pick_offset = np.array([0.0, 0.25, -0.1])
        self._drop_offset = np.array([0.0, 0.00, -0.1])
        self._clearance = 0.00
        self._gripper_close_vel = -0.8
        self._gripper_open_vel = 0.8
        self._gripper_action_time = 1.0
        self._joint_name_candidates = {
            'q1': ['q1', 'Shoulder_Rotation'],
            'q2': ['q2', 'Shoulder_Pitch'],
            'q3': ['q3', 'Elbow'],
            'q4': ['q4', 'Wrist_Pitch'],
            'q5': ['q5', 'Wrist_Roll'],
        }
        self._last_time = self.get_clock().now()

        # numeric kinematics functions (computed once at startup)
        self._fk_func = fk_position_numeric_func()
        self._Jv_func = jv_position_numeric_func()    # used only for IK solving

        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self._marker_pub = self.create_publisher(Marker, 'ee_trace', 10)
        self._trace_marker = Marker()
        self._trace_marker.header.frame_id = 'world'
        self._trace_marker.ns = 'ee_trace'
        self._trace_marker.id = 0
        self._trace_marker.type = Marker.LINE_STRIP
        self._trace_marker.action = Marker.ADD
        self._trace_marker.scale.x = 0.003
        self._trace_marker.color.r = 0.0
        self._trace_marker.color.g = 0.8
        self._trace_marker.color.b = 1.0
        self._trace_marker.color.a = 1.0

        self._targets_marker = Marker()
        self._targets_marker.header.frame_id = 'world'
        self._targets_marker.ns = 'pick_drop_targets'
        self._targets_marker.id = 1
        self._targets_marker.type = Marker.SPHERE_LIST
        self._targets_marker.action = Marker.ADD
        self._targets_marker.scale.x = 0.015
        self._targets_marker.scale.y = 0.015
        self._targets_marker.scale.z = 0.015
        self._targets_marker.color.r = 1.0
        self._targets_marker.color.g = 0.3
        self._targets_marker.color.b = 0.0
        self._targets_marker.color.a = 1.0
        joint_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            joint_state_qos,
        )
        timer_period = 0.04  # seconds
        self._timer = self.create_timer(timer_period, self.timer_callback)

    def _advance_phase(self, t_total):
        """Advance to next phase and initialise its joint-space trajectory parameters."""
        if self._phase_index >= len(self._phases) - 1:
            if self._repeat_sequence:
                self._phase_index = 0
                self._phase_start_time = t_total
                phase = self._phases[self._phase_index]
                self.get_logger().info(
                    f"pickup cycle complete; restarting — phase 0: {phase['label']}")
                self._last_phase_index_logged = self._phase_index
                if phase['type'] == 'joint_move':
                    self._traj_q_start = self._q.copy()
                    self._traj_q_goal  = phase['q_goal'].copy()
                    travel = float(np.max(np.abs(self._traj_q_goal - self._traj_q_start)))
                    self._phase_duration = max(travel / self._qdot_max, 0.5)
                return
            self._sequence_complete = True
            self.get_logger().info('pickup sequence complete; holding position')
            return
        self._phase_index += 1
        self._phase_start_time = t_total
        phase = self._phases[self._phase_index]
        self.get_logger().info(f"phase {self._phase_index}: {phase['label']}")
        self._last_phase_index_logged = self._phase_index
        if phase['type'] == 'joint_move':
            self._traj_q_start = self._q.copy()
            self._traj_q_goal  = phase['q_goal'].copy()
            travel = float(np.max(np.abs(self._traj_q_goal - self._traj_q_start)))
            self._phase_duration = max(travel / self._qdot_max, 0.5)

    def _build_phases(self, p_now):
        """IK-solve all Cartesian waypoints; return a joint-space phase list."""
        pickup = (self._pickup_abs.copy() if self._use_absolute_targets
                  else p_now + self._pick_offset)
        pickup_above = pickup + np.array([0.0, 0.0, self._clearance])

        self.get_logger().info('solving IK for waypoints…')
        q_above = ik_solve_position(pickup_above, self._HOME,
                                    self._fk_func, self._Jv_func)
        q_pick  = ik_solve_position(pickup,       q_above,
                                    self._fk_func, self._Jv_func)

        # FK check: log how close IK solutions are to the targets
        p_above_chk = np.array(self._fk_func(*q_above), dtype=float).reshape(3)
        p_pick_chk  = np.array(self._fk_func(*q_pick),  dtype=float).reshape(3)
        self.get_logger().info(
            f'IK pickup_above: target={np.round(pickup_above,3)} '
            f'FK={np.round(p_above_chk,3)} '
            f'err={np.linalg.norm(p_above_chk-pickup_above)*1e3:.1f}mm')
        self.get_logger().info(
            f'IK pickup:       target={np.round(pickup,3)} '
            f'FK={np.round(p_pick_chk,3)} '
            f'err={np.linalg.norm(p_pick_chk-pickup)*1e3:.1f}mm')

        return [
            {'type': 'joint_move', 'q_goal': q_above,
             'label': 'to pickup above'},
            {'type': 'joint_move', 'q_goal': q_pick,
             'label': 'to pickup'},
            {'type': 'grip',
             'gripper_vel': self._gripper_close_vel,
             'duration':    self._gripper_action_time,
             'label':       'close gripper'},
            {'type': 'joint_move', 'q_goal': q_above.copy(),
             'label': 'lift from pickup'},
            {'type': 'joint_move', 'q_goal': self._HOME.copy(),
             'label': 'return home'},
        ]

    def joint_state_callback(self, msg):
        if not msg.name or not msg.position:
            return

        name_to_idx = {name: index for index, name in enumerate(msg.name)}
        q_new = self._q.copy()

        ordered_keys = ['q1', 'q2', 'q3', 'q4', 'q5']
        for joint_index, key in enumerate(ordered_keys):
            candidates = self._joint_name_candidates[key]
            selected_idx = None
            for candidate in candidates:
                if candidate in name_to_idx:
                    selected_idx = name_to_idx[candidate]
                    break
            if selected_idx is None or selected_idx >= len(msg.position):
                return
            # wrap to [-pi, pi] so accumulated velocity-mode positions
            # are normalised; do NOT clamp – clamping causes state errors
            q_new[joint_index] = float(wrap_to_pi(msg.position[selected_idx]))

        self._q = q_new
        self._have_joint_state = True

    def timer_callback(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        # compute elapsed times
        t_total = (now - self._beginning).nanoseconds * 1e-9
        self._last_time = now

        point = JointTrajectoryPoint()

        # wait for first feedback before moving
        if not self._have_joint_state:
            self.get_logger().warn(
                'waiting for joint_states…', throttle_duration_sec=2.0)
            for _ in range(len(self._HOME)):
                point.velocities.append(0.0)
            point.velocities.append(0.0)
            msg.points = [point]
            self._publisher.publish(msg)
            return

        # --- homing phase: drive joints straight to HOME within limits ---
        if not self._init_done:
            # direct error – no angle wrapping; all limits are < 180° so
            # the shortest path is always the direct one
            q_err = self._HOME - self._q
            if np.max(np.abs(q_err)) < self._init_tol:
                self._init_done = True
                self._beginning = now
                self._home_done_time = now
                self.get_logger().info(
                    'homing complete; settling 1.0s before trajectory start')
            else:
                dq_home = clamp_vec(self._init_kp * q_err, 0.8)
                dq_home = gate_vel_at_limits(self._q, dq_home)
                for rate in dq_home:
                    point.velocities.append(float(rate))
                point.velocities.append(0.0)
                msg.points = [point]
                self._publisher.publish(msg)
                return

        # one-time 1s settle right after homing, then continue as before
        if self._home_done_time is not None:
            settle_elapsed = (now - self._home_done_time).nanoseconds * 1e-9
            if settle_elapsed < self._post_home_settle_sec:
                for _ in range(len(self._HOME)):
                    point.velocities.append(0.0)
                point.velocities.append(0.0)
                msg.points = [point]
                self._publisher.publish(msg)
                return
            self._home_done_time = None

        # desired end-effector twist (vx,vy,vz, wx,wy,wz)
        p_now = np.array(self._fk_func(*self._q), dtype=float).reshape(3)
        if not self._sequence_started:
            # IK-solve all waypoints once, then plan joint-space trajectories
            self._phases = self._build_phases(p_now)
            self._phase_index = 0
            self._phase_start_time = t_total
            self._sequence_started = True
            self._sequence_complete = False
            # init joint-space trajectory for first phase
            phase0 = self._phases[0]
            self._traj_q_start = self._q.copy()
            self._traj_q_goal  = phase0['q_goal'].copy()
            travel0 = float(np.max(np.abs(self._traj_q_goal - self._traj_q_start)))
            self._phase_duration = max(travel0 / self._qdot_max, 0.5)
            self._last_phase_index_logged = 0
            self.get_logger().info(
                f"pickup sequence started; phase 0: {phase0['label']} "
                f"(T={self._phase_duration:.2f} s)")
            # RViz marker: pickup target
            pickup = (self._pickup_abs.copy() if self._use_absolute_targets
                      else p_now + self._pick_offset)
            pick_pt = Point()
            pick_pt.x = float(pickup[0])
            pick_pt.y = float(pickup[1])
            pick_pt.z = float(pickup[2])
            self._targets_marker.points = [pick_pt]
            self._targets_marker.header.stamp = now.to_msg()
            self._marker_pub.publish(self._targets_marker)

        phase = self._phases[self._phase_index]
        if self._phase_index != self._last_phase_index_logged:
            self.get_logger().info(f"phase {self._phase_index}: {phase['label']}")
            self._last_phase_index_logged = self._phase_index
        gripper_vel_cmd = 0.0

        if self._sequence_complete:
            dq = np.zeros(len(self._HOME), dtype=float)
        elif phase['type'] == 'joint_move':
            elapsed = t_total - self._phase_start_time
            alpha = min(elapsed / self._phase_duration, 1.0)
            # minimum-jerk profile in joint space: zero vel at start and end
            s     = 10.0*alpha**3 - 15.0*alpha**4 + 6.0*alpha**5
            s_dot = (30.0*alpha**2 - 60.0*alpha**3 + 30.0*alpha**4) / self._phase_duration
            chord   = self._traj_q_goal - self._traj_q_start
            q_des   = self._traj_q_start + s * chord        # desired joint position
            qdot_ff = s_dot * chord                          # feedforward joint velocity
            dq = qdot_ff + self._kp_joint * (q_des - self._q)  # ff + tracking correction
            dq = clamp_vec(dq, 1.0)
            dq = gate_vel_at_limits(self._q, dq)
            if alpha >= 1.0:
                self._advance_phase(t_total)
            elif (t_total - self._phase_start_time) > self._move_phase_timeout:
                self.get_logger().warn(
                    f"timeout in move phase '{phase['label']}', advancing")
                self._advance_phase(t_total)
        else:
            # grip phase: hold joints still, actuate gripper
            dq = np.zeros(len(self._HOME), dtype=float)
            gripper_vel_cmd = phase['gripper_vel']
            if (t_total - self._phase_start_time) >= phase['duration']:
                self._advance_phase(t_total)

        # form trajectory message using computed joint rates
        for rate in dq:
            point.velocities.append(float(rate))
        point.velocities.append(float(gripper_vel_cmd))

        msg.points = [point]
        self._publisher.publish(msg)

        # publish ee trace marker for RViz
        tp = Point()
        tp.x, tp.y, tp.z = float(p_now[0]), float(p_now[1]), float(p_now[2])
        self._trace_marker.points.append(tp)
        if len(self._trace_marker.points) > 2000:
            self._trace_marker.points.pop(0)
        self._trace_marker.header.stamp = now.to_msg()
        self._marker_pub.publish(self._trace_marker)


def main(args=None):
    rclpy.init(args=args)

    example_traj = ExampleTraj()

    try:
        rclpy.spin(example_traj)
    finally:
        # publish zero velocities on exit so Ctrl+C doesn't leave the robot spinning
        stop_msg = JointTrajectory()
        stop_point = JointTrajectoryPoint()
        stop_point.velocities = [0.0] * 6
        stop_msg.points = [stop_point]
        example_traj._publisher.publish(stop_msg)
        example_traj.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()