import rclpy
import numpy as np
import math
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


# INVERSE KINEMATICS & LIMITS

JOINT_LIMITS_RAD = {
    'q1': (math.radians(-135), math.radians(135)),
    'q2': (math.radians(-90), math.radians(90)),
    'q3': (math.radians(-90), math.radians(90)),
    'q4': (math.radians(-90), math.radians(90)),
    'q5': (math.radians(-180), math.radians(180))
}

def validate_joint_limits(q_vals):
    joint_names = ['q1', 'q2', 'q3', 'q4', 'q5']
    for i, q in enumerate(q_vals):
        name = joint_names[i]
        min_rad, max_rad = JOINT_LIMITS_RAD[name]
        q_norm = (q + math.pi) % (2 * math.pi) - math.pi
        if not (min_rad <= q_norm <= max_rad):
            raise ValueError(f"Joint {name} out of bounds.")
    return q_vals

def compute_inverse_kinematics(X, Y, Z, theta_pitch=None, roll=0.0):
    l2, l3, l4 = 0.1160, 0.1350, 0.1351
    y1, y2 = 0.0452, 0.0306
    z1, z2 = 0.0165, 0.1025
    
    q1 = math.atan2(Y - y1, X)
    
    r_ee = math.sqrt(X**2 + (Y - y1)**2)
    r_target = r_ee - y2
    z_target = Z - (z1 + z2)

    # --- Pitch Sweep Logic ---
    if theta_pitch is not None:
        pitches_to_test = [theta_pitch]
    else:
        pitches = np.arange(-1.57, 1.57, 0.05) 
        # Set preferred pitch to 0.0 (horizontal) for the vertical square.
        # It will tilt away from 0.0 only when necessary to reach a point.
        preferred_pitch = 0.0
        pitches_to_test = sorted(pitches, key=lambda p: abs(p - preferred_pitch))

    for pitch in pitches_to_test:
        try:
            r_w = r_target - l4 * math.cos(pitch)
            z_w = z_target - l4 * math.sin(pitch)

            cos3 = (r_w**2 + z_w**2 - l2**2 - l3**2) / (2 * l2 * l3)
            if cos3 > 1.0 or cos3 < -1.0:
                continue 
            
            cos3 = max(min(cos3, 1.0), -1.0) 
            sin3 = math.sqrt(1 - cos3**2)
            theta_3 = math.atan2(sin3, cos3)

            k1 = l2 + l3 * cos3
            k2 = l3 * sin3
            theta_2 = math.atan2(z_w, r_w) + math.atan2(k2, k1)

            theta_2_off = math.atan2(0.11257, 0.028)
            theta_3_off = math.atan2(0.0052, 0.1349)

            q2 = theta_2 - theta_2_off
            q3 = theta_2_off - theta_3 - theta_3_off
            q4 = pitch - q2 - q3
            q5 = roll

            return validate_joint_limits([q1, q2, q3, q4, q5])
            
        except ValueError:
            continue
            
    raise ValueError("Target is totally out of reach for all safe angles.")


# ---------- Forward Kinematics (numeric) ----------

def rot_z_np(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0, 0],
                     [sa, ca, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def thin_tform_np(pos, rpy):
    rx, ry, rz = rpy
    # roll-pitch-yaw order: x then y then z rotation
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R = Rz.dot(Ry).dot(Rx)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def forward_kinematics(q_vals):
    # use same constants as symbolic derivation
    q1, q2, q3, q4, q5 = q_vals
    T_world_base = thin_tform_np([0, 0, 0], [0, 0, math.pi])
    T_base_shldr = thin_tform_np([0, -0.0452, 0.0165], [0, 0, 0]).dot(rot_z_np(q1))
    T_shldr_upper = thin_tform_np([0, -0.0306, 0.1025], [0, -1.57079, 0]).dot(rot_z_np(q2))
    T_upper_lower = thin_tform_np([0.11257, -0.028, 0], [0, 0, 0]).dot(rot_z_np(q3))
    T_lower_wrist = thin_tform_np([0.0052, -0.1349, 0], [0, 0, math.pi/2]).dot(rot_z_np(q4))
    T_wrist_grip = thin_tform_np([-0.0601, 0, 0], [0, -math.pi/2, 0]).dot(rot_z_np(q5))
    T_grip_center = thin_tform_np([0, 0, 0.075], [0, 0, 0])

    T_fk = (T_world_base.dot(T_base_shldr).dot(T_shldr_upper)
            .dot(T_upper_lower).dot(T_lower_wrist)
            .dot(T_wrist_grip).dot(T_grip_center))
    return T_fk[:3, 3]


# ==========================================
# ROS 2 NODE
# ==========================================
class ExampleTraj(Node):

    def __init__(self):
        super().__init__('example_trajectory')

        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        # marker publisher for end-effector trace
        self._marker_pub = self.create_publisher(Marker, 'fk_ee', 10)
        self._marker = Marker()
        self._marker.header.frame_id = 'world'
        self._marker.ns = 'ee_trace'
        self._marker.id = 0
        self._marker.type = Marker.LINE_STRIP
        self._marker.action = Marker.ADD
        self._marker.scale.x = 0.005
        self._marker.color.r = 1.0
        self._marker.color.g = 0.0
        self._marker.color.b = 0.0
        self._marker.color.a = 1.0
        self._marker.points = []

        self.cycle_time = 20.0 
        
        # --- Horizontal Square Configuration in X-Y plane (reachable) ---
        # The requested y-range with x<0.20 is out of reach at low Z; use safe region instead.
        self.p1 = np.array([ 0.20, 0.10, 0.10])  # Bottom-Left
        self.p2 = np.array([ 0.35, 0.10, 0.10])  # Bottom-Right
        self.p3 = np.array([ 0.35, 0.25, 0.10])  # Top-Right
        self.p4 = np.array([ 0.20, 0.25, 0.10])  # Top-Left
        
        timer_period = 0.04  # 25 Hz
        self._timer = self.create_timer(timer_period, self.timer_callback)

    def get_square_target(self, dt):
        t = dt % self.cycle_time 
        segment_time = self.cycle_time / 4.0
        
        if t < segment_time:
            alpha = t / segment_time
            return (1 - alpha) * self.p1 + alpha * self.p2
        elif t < 2 * segment_time:
            alpha = (t - segment_time) / segment_time
            return (1 - alpha) * self.p2 + alpha * self.p3
        elif t < 3 * segment_time:
            alpha = (t - 2 * segment_time) / segment_time
            return (1 - alpha) * self.p3 + alpha * self.p4
        else:
            alpha = (t - 3 * segment_time) / segment_time
            return (1 - alpha) * self.p4 + alpha * self.p1

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self._beginning).nanoseconds * (1e-9)
        
        target_xyz = self.get_square_target(dt)
        
        try:
            # Change theta_pitch to None so the arm dynamically chooses the best pitch
            q_vals = compute_inverse_kinematics(
                X=target_xyz[0], 
                Y=target_xyz[1], 
                Z=target_xyz[2], 
                theta_pitch=0.0,  # keep end-effector horizontal for XY-plane square
                roll=math.pi/2  # rotate gripper 90 degrees
            )
            
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            
            point = JointTrajectoryPoint()
            point.positions = [float(q) for q in q_vals] + [0.0]  # Keep gripper closed (no oscillation)
            msg.points = [point]

            self._publisher.publish(msg)
            # publish marker trace
            ee_pos = forward_kinematics(q_vals)
            p = Point()
            p.x, p.y, p.z = float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])
            self._marker.points.append(p)
            # limit the number of points to prevent memory issues
            if len(self._marker.points) > 1000:
                self._marker.points.pop(0)
            self._marker.header.stamp = now.to_msg()
            self._marker_pub.publish(self._marker)

        except ValueError as e:
            self.get_logger().warn(f"IK Error at X={target_xyz[0]:.3f}, Y={target_xyz[1]:.3f}, Z={target_xyz[2]:.3f}: {e}")

def main(args=None):
    rclpy.init(args=args)
    example_traj = ExampleTraj()
    
    try:
        rclpy.spin(example_traj)
    except KeyboardInterrupt:
        pass
    finally:
        example_traj.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()