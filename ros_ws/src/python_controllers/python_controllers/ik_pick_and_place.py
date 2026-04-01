import rclpy
from rclpy.node import Node
import numpy as np
import math

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ==========================================
# RELATIVE INVERSE KINEMATICS FUNCTION
# ==========================================
def get_offset_angles(base_place_angles, z_offset_meters):
    """
    Takes known physical joint angles, finds their theoretical 2D Cartesian plane,
    adds a Z offset, and calculates the new joint angles required to reach that height
    while keeping the horizontal extension and tool pitch locked perfectly in place.
    """
    q1_base, q2_base, q3_base, q4_base, q5_base, gripper = base_place_angles
    
    if z_offset_meters == 0.0:
        return base_place_angles # No change required
        
    # Apply Hardware Offsets---
    # Physical_Angle = Software_Angle - Offset
    off_q2 = -0.135
    off_q3 = 0.021
    off_q4 = -0.025
    
    q2_true = q2_base - off_q2
    q3_true = q3_base - off_q3
    q4_true = q4_base - off_q4
    
    # --- Kinematic Link Lengths & URDF Offsets ---
    l2 = 0.1160
    l3 = 0.1350
    theta_2_off = math.atan2(0.11257, 0.028)
    theta_3_off = math.atan2(0.0052, 0.1349)
    
    # --- Forward Kinematics (Using TRUE angles) ---
    theta_2 = q2_true + theta_2_off
    theta_3 = theta_2_off - q3_true - theta_3_off
    pitch = q2_true + q3_true + q4_true
    
    r_w = l2 * math.cos(theta_2) + l3 * math.cos(theta_2 - theta_3)
    z_w = l2 * math.sin(theta_2) + l3 * math.sin(theta_2 - theta_3)
    
    # --- Add the requested Z height ---
    z_w_new = z_w + z_offset_meters
    
    # --- Inverse Kinematics for the new height ---
    cos3 = (r_w**2 + z_w_new**2 - l2**2 - l3**2) / (2 * l2 * l3)
    if cos3 > 1.0 or cos3 < -1.0:
        raise ValueError(f"Target height offset of +{z_offset_meters}m is out of physical reach.")
        
    sin3 = math.sqrt(1 - cos3**2)
    theta_3_new = math.atan2(sin3, cos3)
    
    k1 = l2 + l3 * cos3
    k2 = l3 * sin3
    theta_2_new = math.atan2(z_w_new, r_w) + math.atan2(k2, k1)
    
    # --- Map back to TRUE physical joint angles ---
    q2_new_true = theta_2_new - theta_2_off
    q3_new_true = theta_2_off - theta_3_new - theta_3_off
    q4_new_true = pitch - q2_new_true - q3_new_true
    
    # --- Convert TRUE angles back to Software Angles ---
    q2_new = q2_new_true + off_q2
    q3_new = q3_new_true + off_q3
    q4_new = q4_new_true + off_q4
    
    # Note: q1 and q5 are passed through directly because they don't affect the 2D plane lift
    return [q1_base, q2_new, q3_new, q4_new, q5_base, gripper]

# ==========================================
# PICK AND PLACE NODE
# ==========================================
class PickAndPlaceEduBot(Node):

    def __init__(self):
        super().__init__('pick_and_place_edubot')
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        # ---------------------------------------------------------
        # Hardware Configuration
        # ---------------------------------------------------------
        GRIPPER_OPEN = 0.6
        GRIPPER_CLOSED = 0.3

        # Hardcoded Home and Pick positions
        HOME       = [0.02141, 0.2929, -0.5583, -1.6613, -1.6321, GRIPPER_OPEN]
        PRE_GRASP  = [0.042141, 0.2929, -0.5583, -1.6613, -1.6321, GRIPPER_OPEN]
        GRASP_0    = [0.0241, 0.06289, -0.9480, -1.13054, -1.5830, GRIPPER_CLOSED]
        POST_GRASP = [0.02141, 0.2929, -0.5583, -1.6613, -1.6321, GRIPPER_CLOSED]

        # ---------------------------------------------------------
        # Dynamic Stacking via Relative IK
        # ---------------------------------------------------------
        PLACE_BASE = 0.9100
        PLACE_TILT = -0.7480 # Vertical tilt
        
        PLACE = [PLACE_BASE, 0.2715, -1.1137, -1.0324, PLACE_TILT, GRIPPER_CLOSED]

        stack_targets = []
        
        # Generate target configurations for 8 blocks (+2cm per block)
        for i in range(8):
            z_offset = i * 0.020 # From 0.00m up to 0.14m
            try:
                new_angles = get_offset_angles(PLACE, z_offset)
                stack_targets.append(new_angles)
                
                # Print the calculated angles to the terminal for debugging
                angles_str = ", ".join([f"{q:.4f}" for q in new_angles])
                self.get_logger().info(f"Calculated IK for Block {i+1} (+{z_offset}m): [{angles_str}]")
                
            except ValueError as e:
                self.get_logger().error(str(e))
                stack_targets.append(HOME) # Fallback if out of reach

        # Calculate Universal Clearances (Hovering 15cm above PLACE_1)
        try:
            hover_angles = get_offset_angles(PLACE, 0.150)
            UNIVERSAL_PRE_PLACE  = hover_angles[:5] + [GRIPPER_CLOSED]
            UNIVERSAL_POST_PLACE = hover_angles[:5] + [GRIPPER_OPEN]
        except ValueError:
            self.get_logger().warn("Hover height out of reach, using safe approximation.")
            UNIVERSAL_PRE_PLACE  = [PLACE_BASE, 0.0000, -0.8713, -1.3898, PLACE_TILT, GRIPPER_CLOSED]
            UNIVERSAL_POST_PLACE = [PLACE_BASE, 0.0000, -0.8713, -1.3898, PLACE_TILT, GRIPPER_OPEN]

        
        # ---------------------------------------------------------
        # OPTIMIZED Sequence Generation
        # ---------------------------------------------------------
        self.sequence = [(HOME, 3.0, "Moving to Home")] # Can be skipped to save time

        for i, target_place in enumerate(stack_targets):
            block_num = i + 1
            
            # Calculate a Dynamic Hover (Only 4cm above THIS specific block's drop point)
            try:
                hover_angles = get_offset_angles(target_place, 0.040)
            except ValueError:
                hover_angles = get_offset_angles(target_place, 0.010) # Fallback to 1cm if max reach hit
                
            DYNAMIC_PRE_PLACE  = hover_angles[:5] + [GRIPPER_CLOSED]
            
            # Overlap the Gripper opening with the retreat

            DYNAMIC_POST_PLACE = hover_angles[:5] + [GRIPPER_OPEN]

            self.sequence.extend([
                # --- PICK PHASE ---
                (PRE_GRASP,  1.3, f"[Block {block_num}] Sweeping to pick zone"),
                (GRASP_0,    1.2, f"[Block {block_num}] Plunge"),
                (POST_GRASP, 0.9, f"[Block {block_num}] Bite & Lift"),
                
                # --- PLACE PHASE ---
                (DYNAMIC_PRE_PLACE,  1.7, f"[Block {block_num}] Sweeping to Stack"),
                (target_place,       1.7, f"[Block {block_num}] Gentle Drop"),
                (DYNAMIC_POST_PLACE, 1.0, f"[Block {block_num}] Fast Retreat & Release")
            ])
            
        # Return home after stacking is complete
        self.sequence.append((HOME, 2.0, "Time's up! Returning Home"))
    
        # State tracking
        self.current_step = 0
        self.start_config = HOME 
        self.start_time = self.get_clock().now()

        # Timer running at 50 Hz
        self._timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info(f"Starting Sequence: {self.sequence[0][2]}")


    def smooth_step(self, x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def timer_callback(self):
        if self.current_step >= len(self.sequence):
            return

        now = self.get_clock().now()
        target_config, duration, label = self.sequence[self.current_step]
        
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        if elapsed >= duration:
            self.get_logger().info(f"Completed: {label}")
            self.current_step += 1
            
            if self.current_step >= len(self.sequence):
                self.get_logger().info("=== Pick and Place Complete! ===")
                self._timer.cancel()
                return
            
            self.start_config = target_config
            self.start_time = now
            next_label = self.sequence[self.current_step][2]
            self.get_logger().info(f"Starting Sequence: {next_label}")
            return

        alpha = elapsed / duration
        alpha = self.smooth_step(alpha)

        current_positions = []
        for i in range(6):
            start_val = self.start_config[i]
            target_val = target_config[i]
            current_positions.append(start_val + alpha * (target_val - start_val))

        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in current_positions]
        msg.points = [point]
        
        self._publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceEduBot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Sequence interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()