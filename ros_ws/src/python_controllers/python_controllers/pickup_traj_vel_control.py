import threading

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
except ImportError:
    rclpy = None
    class Node:
        def __init__(self, *args, **kwargs):
            pass

import numpy as np
import sympy as sp

try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
    from builtin_interfaces.msg import Duration
except ImportError:
    class JointTrajectory:
        def __init__(self):
            self.header = type('h', (), {'stamp': None})
            self.points = []
    class JointTrajectoryPoint:
        def __init__(self):
            self.velocities = []
            self.positions  = []
    class Duration:
        def __init__(self, sec=0, nanosec=0):
            self.sec = sec
            self.nanosec = nanosec

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
    T[:3,3]  = sp.Matrix(pos)
    return T

# ----------------------------
# Symbolic forward kinematics
# ----------------------------
def get_symbolic_T_world_gc():
    q1,q2,q3,q4,q5 = sp.symbols('q1 q2 q3 q4 q5', real=True)

    T_world_base = thin_tform([0,0,0],            [0, 0,  sp.pi])
    T_base_sh    = thin_tform([0,-0.0452,0.0165], [0, 0,  0])       * rot_z_4(q1)
    T_sh_ua      = thin_tform([0,-0.0306,0.1025], [0,-1.57079,0])   * rot_z_4(q2)
    T_ua_la      = thin_tform([0.11257,-0.028,0], [0, 0,  0])       * rot_z_4(q3)
    T_la_wr      = thin_tform([0.0052,-0.1349,0], [0, 0,  sp.pi/2]) * rot_z_4(q4)
    T_wr_gr      = thin_tform([-0.0601,0,0],      [0,-sp.pi/2,0])   * rot_z_4(q5)
    T_gr_gc      = thin_tform([0,0,0.075],        [0, 0,  0])

    T = T_world_base*T_base_sh*T_sh_ua*T_ua_la*T_la_wr*T_wr_gr*T_gr_gc
    return T, (q1,q2,q3,q4,q5)

# ----------------------------
# Symbolic Jacobian
# ----------------------------
def get_symbolic_jacobian():
    T, q = get_symbolic_T_world_gc()
    p    = T[:3, 3]
    Jv   = p.jacobian(sp.Matrix(q))
    z    = sp.Matrix([0,0,1])
    q1,q2,q3,q4,q5 = q

    Twb = thin_tform([0,0,0],           [0,0,sp.pi])
    Tbs = thin_tform([0,-0.0452,0.0165],[0,0,0])
    Tsu = thin_tform([0,-0.0306,0.1025],[0,-1.57079,0])
    Tul = thin_tform([0.11257,-0.028,0],[0,0,0])
    Tlw = thin_tform([0.0052,-0.1349,0],[0,0,sp.pi/2])
    Twg = thin_tform([-0.0601,0,0],     [0,-sp.pi/2,0])

    T0 = Twb
    T1 = T0*(Tbs*rot_z_4(q1))
    T2 = T1*(Tsu*rot_z_4(q2))
    T3 = T2*(Tul*rot_z_4(q3))
    T4 = T3*(Tlw*rot_z_4(q4))

    Jw = sp.Matrix.hstack(
        (T0*Tbs)[:3,:3]*z, (T1*Tsu)[:3,:3]*z,
        (T2*Tul)[:3,:3]*z, (T3*Tlw)[:3,:3]*z,
        (T4*Twg)[:3,:3]*z,
    )
    return sp.Matrix.vstack(Jv, Jw), Jv, Jw, q

# ----------------------------
# Lambdified numeric functions
# ----------------------------
def fk_position_numeric_func():
    T_sym, q_syms = get_symbolic_T_world_gc()
    return sp.lambdify(q_syms, T_sym[:3, 3], modules="numpy")

def jv_position_numeric_func():
    T_sym, q_syms = get_symbolic_T_world_gc()
    p_sym = T_sym[:3, 3]
    return sp.lambdify(q_syms, p_sym.jacobian(sp.Matrix(q_syms)), modules="numpy")

# ----------------------------
# Joint limits & helpers
# ----------------------------
JOINT_LIMITS_RAD = np.array([
    [np.deg2rad(-135.0), np.deg2rad(135.0)],
    [np.deg2rad(-120.0), np.deg2rad(120.0)],
    [np.deg2rad(-120.0), np.deg2rad(120.0)],
    [np.deg2rad(-100.0), np.deg2rad(100.0)],
    [np.deg2rad(-180.0), np.deg2rad(180.0)],
], dtype=float)

def wrap_to_pi(a):
    return (a + np.pi) % (2.0*np.pi) - np.pi

def clamp_q_to_limits(q):
    return np.clip(np.asarray(q, dtype=float),
                   JOINT_LIMITS_RAD[:,0], JOINT_LIMITS_RAD[:,1])

def gate_vel_at_limits(q, dq, margin=np.deg2rad(1.0)):
    """Zero out velocity components that would drive a joint past its limit."""
    q   = np.asarray(q,  dtype=float)
    dq  = np.asarray(dq, dtype=float)
    out = dq.copy()
    lo  = JOINT_LIMITS_RAD[:, 0] + margin
    hi  = JOINT_LIMITS_RAD[:, 1] - margin
    out = np.where((q <= lo) & (out < 0.0), 0.0, out)
    out = np.where((q >= hi) & (out > 0.0), 0.0, out)
    return out

def validate_joint_limits(q_vals):
    q = wrap_to_pi(np.asarray(q_vals, dtype=float))
    for i, v in enumerate(q):
        lo, hi = JOINT_LIMITS_RAD[i]
        if not (lo <= v <= hi):
            raise ValueError(
                f"q{i+1} = {np.rad2deg(v):.1f}° outside "
                f"[{np.rad2deg(lo):.1f}, {np.rad2deg(hi):.1f}]°")
    return q

def ik_solve_position(p_target, q_init, fk_func, jv_func,
                      max_iter=300, tol=8e-4, lam=2e-2):
    q = np.array(q_init, dtype=float)
    for _ in range(max_iter):
        err = p_target - np.array(fk_func(*q), dtype=float).reshape(3)
        if np.linalg.norm(err) < tol:
            break
        Jv = np.array(jv_func(*q), dtype=float)
        q  = clamp_q_to_limits(
                q + np.clip(
                    Jv.T @ np.linalg.inv(Jv @ Jv.T + lam**2 * np.eye(3)) @ err,
                    -0.3, 0.3))
    return q

def compute_inverse_kinematics_consistent(X, Y, Z, theta_pitch=None,
                                           roll=0.0, preferred_pitch=0.0):
    l2, l3, l4     = 0.1160, 0.1350, 0.1351
    y1, y2, z1, z2 = 0.0452, 0.0306, 0.0165, 0.1025

    q1       = np.arctan2(Y - y1, X) - np.pi/2.0
    r_target = np.sqrt(X**2 + (Y - y1)**2) - y2
    z_target = Z - (z1 + z2)

    pitches = ([float(theta_pitch)] if theta_pitch is not None
               else sorted(np.arange(-1.57, 1.57, 0.05),
                           key=lambda v: abs(v - preferred_pitch)))
    last_err = None
    for pitch in pitches:
        try:
            rw = r_target - l4*np.cos(pitch)
            zw = z_target - l4*np.sin(pitch)
            c3 = (rw**2 + zw**2 - l2**2 - l3**2) / (2.0*l2*l3)
            if not (-1.0 <= c3 <= 1.0):
                continue
            s3  = np.sqrt(1.0 - c3**2)
            t3  = np.arctan2(s3, c3)
            t2  = np.arctan2(zw, rw) + np.arctan2(l3*s3, l2 + l3*c3)
            off2 = np.arctan2(0.11257, 0.028)
            off3 = np.arctan2(0.0052,  0.1349)
            q2  = t2 - off2
            q3  = off2 - t3 - off3
            q4  = pitch - q2 - q3
            return validate_joint_limits([q1, q2, q3, q4, roll])
        except ValueError as e:
            last_err = e
    raise ValueError(f'IK failed: {last_err}')


class ExampleTrajVelocity(Node):
    """
    Same precomputed trajectory as ExampleTraj but streams VELOCITY commands
    (finite differences of position samples) instead of position commands.

    Homing still uses position interpolation (smooth S-curve).
    During trajectory streaming, velocity = (q[k] - q[k-1]) / TIMER_DT,
    plus a proportional correction on the real joint error — giving the
    smoothness of precomputed trajectories with the tracking robustness of
    the original velocity-mode controller.
    """

    TIMER_DT     = 0.04   # seconds — must match create_timer period
    VEL_LIMIT    = 1.2    # rad/s hard clamp on any joint velocity
    KP_TRACKING  = 1.5    # gain on real-state error correction during streaming

    def __init__(self):
        super().__init__('example_traj_velocity')

        self._HOME = clamp_q_to_limits(np.array([
            np.deg2rad(0), np.deg2rad(60),
            np.deg2rad(-45), np.deg2rad(-85),
            np.deg2rad(0)
        ], dtype=float))

        self._q                = self._HOME.copy()
        self._have_joint_state = False
        self._init_done        = False

        # homing — smooth position interpolation
        self._home_start_q     = None
        self._home_start_time  = None
        self._home_duration    = 3.0
        self._post_home_settle = 1.0
        self._home_done_time   = None

        # precomputed trajectory
        # Each sample: (q[5], gripper_pos)  — positions only, velocities derived
        self._traj_samples     = []
        self._traj_index       = 0
        self._traj_ready       = False
        self._repeat_sequence  = True
        self._compute_lock     = threading.Lock()

        # trajectory shaping
        self._qdot_max          = 0.5   # rad/s — drives segment durations
        self._min_move_duration = 0.05  # s

        # gripper velocity limits
        self._gripper_vel_limit  = 2.0   # rad/s (or appropriate unit)
        self._gripper_open_pos   = 0.6
        self._gripper_closed_pos = 0.3
        self._gripper_open_time  = 0.60
        self._gripper_close_time = 0.60

        # pick-and-place geometry
        self._use_absolute_targets = False
        self._pickup_abs   = np.array([0.3,   0.1,   0.0],  dtype=float)
        self._dropoff_abs  = np.array([-0.2,  0.10,  0.0],  dtype=float)
        self._pick_offset  = np.array([0.0,   0.10, -0.06], dtype=float)
        self._drop_offset  = np.array([0.10,  0.10, -0.06], dtype=float)
        self._num_pickups      = 3
        self._drop_height_step = 0.03
        self._clearance        = 0.05
        self._pickup_pitch     = -np.pi / 2   # straight down; cascades shallower if needed

        self._joint_name_candidates = {
            'q1': ['q1', 'Shoulder_Rotation'],
            'q2': ['q2', 'Shoulder_Pitch'],
            'q3': ['q3', 'Elbow'],
            'q4': ['q4', 'Wrist_Pitch'],
            'q5': ['q5', 'Wrist_Roll'],
        }

        self._fk_func = fk_position_numeric_func()
        self._Jv_func = jv_position_numeric_func()

        self._beginning  = self.get_clock().now()
        self._publisher  = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
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

        self._targets_marker = Marker()
        self._targets_marker.header.frame_id = 'world'
        self._targets_marker.ns     = 'pick_drop_targets'
        self._targets_marker.id     = 1
        self._targets_marker.type   = Marker.SPHERE_LIST
        self._targets_marker.action = Marker.ADD
        self._targets_marker.scale.x = 0.015
        self._targets_marker.scale.y = 0.015
        self._targets_marker.scale.z = 0.015
        self._targets_marker.color.r = 1.0
        self._targets_marker.color.g = 0.3
        self._targets_marker.color.b = 0.0
        self._targets_marker.color.a = 1.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, qos)
        self._timer = self.create_timer(self.TIMER_DT, self.timer_callback)

    # ── Easing curves ─────────────────────────────────────────────────────────
    @staticmethod
    def smooth_step(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def min_jerk(alpha: float) -> float:
        return 10*alpha**3 - 15*alpha**4 + 6*alpha**5

    # ── Publish helpers ───────────────────────────────────────────────────────
    def _publish_positions(self, positions_6: list):
        """Position command — used only during homing."""
        msg   = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()
        point.positions       = [float(v) for v in positions_6]
        point.time_from_start = Duration(sec=0,
                                         nanosec=int(self.TIMER_DT * 1e9))
        msg.points = [point]
        self._publisher.publish(msg)

    def _publish_velocities(self, velocities_6: list):
        """Velocity command — used during trajectory streaming."""
        msg   = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()
        point.velocities = [float(v) for v in velocities_6]
        msg.points = [point]
        self._publisher.publish(msg)

    def _publish_zero_velocity(self):
        self._publish_velocities([0.0] * 6)

    # ── IK solvers ────────────────────────────────────────────────────────────
    def _solve_numeric(self, p_target, q_seed):
        q    = ik_solve_position(np.asarray(p_target, dtype=float),
                                  np.asarray(q_seed,   dtype=float),
                                  self._fk_func, self._Jv_func)
        q    = clamp_q_to_limits(q)
        q[4] = float(self._HOME[4])
        return validate_joint_limits(q)

    def _solve_pickup(self, p_target, q_seed):
        pitch = self._pickup_pitch
        for attempt in [pitch, pitch * 0.85, pitch * 0.70]:
            try:
                return compute_inverse_kinematics_consistent(
                    float(p_target[0]), float(p_target[1]), float(p_target[2]),
                    theta_pitch=attempt, roll=float(self._HOME[4]))
            except ValueError:
                continue
        self.get_logger().warn(
            f'analytic pickup IK failed for {np.round(p_target,3)}; '
            'using numeric fallback')
        return self._solve_numeric(p_target, q_seed)

    # ── Phase list builder ────────────────────────────────────────────────────
    def _build_phases(self, p_now):
        pickup_base  = (self._pickup_abs.copy()  if self._use_absolute_targets
                        else p_now + self._pick_offset)
        dropoff_base = (self._dropoff_abs.copy() if self._use_absolute_targets
                        else p_now + self._drop_offset)

        dropoffs = [
            dropoff_base + np.array([0.0, 0.0, i * self._drop_height_step])
            for i in range(self._num_pickups)
        ]

        self.get_logger().info('IK: solving pickup waypoints…')
        pickup_above_pos = pickup_base + np.array([0.0, 0.0, self._clearance])
        q_seed      = self._q.copy()
        q_above_ref = self._solve_pickup(pickup_above_pos, q_seed)
        q_pick_ref  = self._solve_pickup(pickup_base,      q_above_ref)

        p_chk = np.array(self._fk_func(*q_pick_ref), dtype=float).reshape(3)
        self.get_logger().info(
            f'pickup IK (shared): target={np.round(pickup_base,3)} '
            f'err={np.linalg.norm(p_chk - pickup_base)*1e3:.1f} mm')

        phases = [
            {'type': 'grip', 'gripper_target': self._gripper_open_pos,
             'duration': self._gripper_open_time, 'label': 'open gripper'},
        ]

        for i, dropoff in enumerate(dropoffs, start=1):
            dropoff_above = dropoff + np.array([0.0, 0.0, self._clearance])
            transit_wp    = (0.5*(pickup_above_pos + dropoff_above)
                             + np.array([0.0, 0.0, 0.03]))

            self.get_logger().info(f'IK: solving dropoff [{i}]…')
            q_drop_above = self._solve_numeric(dropoff_above, q_pick_ref)
            q_drop       = self._solve_numeric(dropoff,       q_drop_above)
            q_transit    = self._solve_numeric(transit_wp,    q_drop)

            p_drop_chk = np.array(self._fk_func(*q_drop), dtype=float).reshape(3)
            self.get_logger().info(
                f'drop[{i}] IK: target={np.round(dropoff,3)} '
                f'err={np.linalg.norm(p_drop_chk - dropoff)*1e3:.1f} mm')

            phases += [
                {'type': 'joint_move', 'q_goal': q_above_ref.copy(),
                 'label': f'to pickup above (block {i})'},
                {'type': 'joint_move', 'q_goal': q_pick_ref.copy(),
                 'label': f'to pickup (block {i})'},
                {'type': 'grip', 'gripper_target': self._gripper_closed_pos,
                 'duration': self._gripper_close_time,
                 'label': f'close gripper (block {i})'},
                {'type': 'joint_move', 'q_goal': q_above_ref.copy(),
                 'label': f'lift from pickup (block {i})'},
                {'type': 'joint_move', 'q_goal': q_drop_above,
                 'label': f'to dropoff above (block {i})'},
                {'type': 'joint_move', 'q_goal': q_drop,
                 'label': f'to dropoff (block {i})'},
                {'type': 'grip', 'gripper_target': self._gripper_open_pos,
                 'duration': self._gripper_open_time,
                 'label': f'open at dropoff (block {i})'},
                {'type': 'joint_move', 'q_goal': q_drop_above.copy(),
                 'label': f'lift from dropoff (block {i})'},
                {'type': 'joint_move', 'q_goal': q_transit,
                 'label': f'transit (block {i})'},
            ]

        # smooth return to HOME — prevents jump on cycle restart
        phases.append({'type': 'joint_move', 'q_goal': self._HOME.copy(),
                       'label': 'return HOME'})
        return phases

    # ── Trajectory precomputer ────────────────────────────────────────────────
    def _precompute_trajectory(self, phases, q_actual):
        """
        Returns a list of (q[5], gripper_pos) position samples,
        starting from q_actual (real robot state at homing end).
        Velocities are derived on-the-fly in timer_callback via finite difference.
        """
        dt      = self.TIMER_DT
        samples = []

        q_cur = q_actual.copy()
        g_cur = self._gripper_open_pos

        for phase in phases:
            if phase['type'] == 'joint_move':
                q_start  = q_cur.copy()
                q_goal   = phase['q_goal'].copy()
                travel   = float(np.max(np.abs(q_goal - q_start)))
                duration = max(travel / self._qdot_max, self._min_move_duration)
                n        = max(int(np.ceil(duration / dt)), 1)
                for k in range(1, n + 1):
                    s = self.min_jerk(k / n)
                    samples.append((q_start + s*(q_goal - q_start), g_cur))
                q_cur = q_goal.copy()

            else:  # grip
                g_start = g_cur
                g_goal  = float(phase['gripper_target'])
                n       = max(int(np.ceil(phase['duration'] / dt)), 1)
                for k in range(1, n + 1):
                    s = self.smooth_step(k / n)
                    samples.append((q_cur.copy(),
                                    g_start + s*(g_goal - g_start)))
                g_cur = g_goal

        total_s = len(samples) * dt
        self.get_logger().info(
            f'trajectory precomputed: {len(samples)} samples '
            f'≈ {total_s:.1f} s per cycle')
        return samples

    # ── Background compute thread ─────────────────────────────────────────────
    def _compute_in_background(self, p_now, q_actual):
        try:
            phases  = self._build_phases(p_now)
            samples = self._precompute_trajectory(phases, q_actual)
            with self._compute_lock:
                self._traj_samples = samples
                self._traj_index   = 0
                self._traj_ready   = True
            self.get_logger().info('background compute done — ready to stream')
        except Exception as e:
            self.get_logger().error(f'background compute failed: {e}')

    # ── RViz markers ─────────────────────────────────────────────────────────
    def _publish_rviz_markers(self, p_now):
        pickup_base  = (self._pickup_abs.copy()  if self._use_absolute_targets
                        else p_now + self._pick_offset)
        dropoff_base = (self._dropoff_abs.copy() if self._use_absolute_targets
                        else p_now + self._drop_offset)

        pts = []
        pt = Point()
        pt.x, pt.y, pt.z = float(pickup_base[0]), float(pickup_base[1]), float(pickup_base[2])
        pts.append(pt)
        for i in range(self._num_pickups):
            d  = dropoff_base + np.array([0., 0., i*self._drop_height_step])
            pt = Point()
            pt.x, pt.y, pt.z = float(d[0]), float(d[1]), float(d[2])
            pts.append(pt)

        self._targets_marker.points       = pts
        self._targets_marker.header.stamp = self.get_clock().now().to_msg()
        self._marker_pub.publish(self._targets_marker)

    # ── Joint state feedback ──────────────────────────────────────────────────
    def joint_state_callback(self, msg):
        if not msg.name or not msg.position:
            return
        idx_map = {n: i for i, n in enumerate(msg.name)}
        q_new   = self._q.copy()
        for ji, key in enumerate(['q1','q2','q3','q4','q5']):
            for cand in self._joint_name_candidates[key]:
                if cand in idx_map and idx_map[cand] < len(msg.position):
                    q_new[ji] = float(wrap_to_pi(msg.position[idx_map[cand]]))
                    break
            else:
                return
        self._q                = q_new
        self._have_joint_state = True

    # ── Main control loop ─────────────────────────────────────────────────────
    def timer_callback(self):
        now = self.get_clock().now()

        # 1 ── wait for first joint feedback ───────────────────────────────────
        if not self._have_joint_state:
            self.get_logger().warn('waiting for joint_states…',
                                   throttle_duration_sec=2.0)
            self._publish_positions(list(self._HOME) + [self._gripper_open_pos])
            return

        # 2 ── homing — smooth S-curve, position control ───────────────────────
        if not self._init_done:
            if self._home_start_q is None:
                self._home_start_q    = self._q.copy()
                self._home_start_time = now
                self.get_logger().info(
                    f'homing: S-curve to HOME over {self._home_duration:.1f} s')

            elapsed  = (now - self._home_start_time).nanoseconds * 1e-9
            alpha    = min(elapsed / self._home_duration, 1.0)
            s        = self.smooth_step(alpha)
            q_interp = self._home_start_q + s*(self._HOME - self._home_start_q)

            self._publish_positions(list(q_interp) + [self._gripper_open_pos])

            if alpha >= 1.0:
                self._init_done  = True
                q_actual = self._q.copy()
                p_now    = np.array(self._fk_func(*q_actual),
                                    dtype=float).reshape(3)
                self._publish_rviz_markers(p_now)
                self._home_done_time = now
                threading.Thread(
                    target=self._compute_in_background,
                    args=(p_now, q_actual),
                    daemon=True).start()
                self.get_logger().info(
                    'homing done; computing trajectory in background…')
            return

        # 3 ── settle + wait for background compute ────────────────────────────
        if self._home_done_time is not None:
            elapsed = (now - self._home_done_time).nanoseconds * 1e-9
            if elapsed < self._post_home_settle or not self._traj_ready:
                self._publish_zero_velocity()
                return
            self._home_done_time = None
            self.get_logger().info('streaming velocity trajectory…')

        # 4 ── stream velocity commands from precomputed position samples ───────
        with self._compute_lock:
            if not self._traj_samples:
                self._publish_zero_velocity()
                return

            if self._traj_index >= len(self._traj_samples):
                if self._repeat_sequence:
                    self._traj_index = 0
                    self.get_logger().info('cycle complete — restarting')
                else:
                    self._publish_zero_velocity()
                    return

            q_des, g_des = self._traj_samples[self._traj_index]

            # finite-difference feedforward velocity from precomputed positions
            if self._traj_index > 0:
                q_prev, g_prev = self._traj_samples[self._traj_index - 1]
            else:
                q_prev, g_prev = q_des, g_des   # first sample: zero ff vel

            self._traj_index += 1

        dq_ff  = (q_des - q_prev) / self.TIMER_DT
        dg_ff  = (g_des - g_prev) / self.TIMER_DT

        # proportional correction on real joint error — the key difference
        # from pure open-loop: if the robot lags, this pulls it back on track
        q_err  = q_des - self._q
        dq_cmd = dq_ff + self.KP_TRACKING * q_err

        # clamp and gate at limits
        dq_cmd = np.clip(dq_cmd, -self.VEL_LIMIT, self.VEL_LIMIT)
        dq_cmd = gate_vel_at_limits(self._q, dq_cmd)
        dg_cmd = np.clip(dg_ff, -self._gripper_vel_limit, self._gripper_vel_limit)

        self._publish_velocities(list(dq_cmd) + [float(dg_cmd)])

        # EE trace for RViz
        p_now = np.array(self._fk_func(*self._q), dtype=float).reshape(3)
        tp = Point()
        tp.x, tp.y, tp.z = float(p_now[0]), float(p_now[1]), float(p_now[2])
        self._trace_marker.points.append(tp)
        if len(self._trace_marker.points) > 2000:
            self._trace_marker.points.pop(0)
        self._trace_marker.header.stamp = now.to_msg()
        self._marker_pub.publish(self._trace_marker)


def main(args=None):
    rclpy.init(args=args)
    node = ExampleTrajVelocity()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero_velocity()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
