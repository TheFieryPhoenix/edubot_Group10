#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ==========================================
# 1. RELATIVE INVERSE KINEMATICS FUNCTION
# ==========================================
# 1. RELATIVE INVERSE KINEMATICS FUNCTION
# ==========================================
def get_offset_angles(base_place_angles, z_offset_meters, r_offset_meters=0.0):
    """
    Calculates new joint angles for a Z (height) and R (horizontal reach) offset,
    keeping the tool pitch locked perfectly in place.
    """
    q1_base, q2_base, q3_base, q4_base, q5_base, gripper = base_place_angles
    
    if z_offset_meters == 0.0 and r_offset_meters == 0.0:
        return base_place_angles
        
    # --- Step 1: Apply Hardware Offsets ---
    off_q2, off_q3, off_q4 = -0.135, 0.021, -0.025
    q2_true = q2_base - off_q2
    q3_true = q3_base - off_q3
    q4_true = q4_base - off_q4
    
    # --- Kinematic Link Lengths & URDF Offsets ---
    l2, l3 = 0.1160, 0.1350
    theta_2_off = math.atan2(0.11257, 0.028)
    theta_3_off = math.atan2(0.0052, 0.1349)
    
    # --- Step 2: Forward Kinematics ---
    theta_2 = q2_true + theta_2_off
    theta_3 = theta_2_off - q3_true - theta_3_off
    pitch = q2_true + q3_true + q4_true
    
    r_w = l2 * math.cos(theta_2) + l3 * math.cos(theta_2 - theta_3)
    z_w = l2 * math.sin(theta_2) + l3 * math.sin(theta_2 - theta_3)
    
    # --- Step 3: Add the requested Z and R offsets ---
    z_w_new = z_w + z_offset_meters
    r_w_new = r_w + r_offset_meters  # <--- NEW: Adjusts horizontal reach
    
    # --- Step 4: Inverse Kinematics ---
    cos3 = (r_w_new**2 + z_w_new**2 - l2**2 - l3**2) / (2 * l2 * l3)
    if cos3 > 1.0 or cos3 < -1.0:
        raise ValueError(f"Target offset Z:+{z_offset_meters}m, R:{r_offset_meters}m is out of reach.")
        
    sin3 = math.sqrt(1 - cos3**2)
    theta_3_new = math.atan2(sin3, cos3)
    
    k1 = l2 + l3 * cos3
    k2 = l3 * sin3
    theta_2_new = math.atan2(z_w_new, r_w_new) + math.atan2(k2, k1)
    
    # --- Step 5 & 6: Map back to Software Angles ---
    q2_new_true = theta_2_new - theta_2_off
    q3_new_true = theta_2_off - theta_3_new - theta_3_off
    q4_new_true = pitch - q2_new_true - q3_new_true
    
    q2_new = q2_new_true + off_q2
    q3_new = q3_new_true + off_q3
    q4_new = q4_new_true + off_q4
    
    return [q1_base, q2_new, q3_new, q4_new, q5_base, gripper]

# ==========================================
# 2. PICK AND PLACE NODE
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
        PLACE_TILT = -0.7480
        
        # Your confirmed "Perfect" base standard
        PLACE_1 = [PLACE_BASE, 0.2715, -1.1137, -1.0324, PLACE_TILT, GRIPPER_CLOSED]

        stack_targets = []
        
        # Generate target configurations for 8 blocks (+2cm per block)
# Generate target configurations for 8 blocks (+2cm per block)
        for i in range(8):
            z_offset = i * 0.020 
            r_offset = 0.003 # Default: perfectly straight up
            
            # --- MANUALLY TUNE SPECIFIC BLOCKS HERE ---
            if i == 5: # Block 7
                r_offset = -0.030 # Pull 5mm backwards (closer to robot)

            elif i == 6: # Block 8
                r_offset = -0.030 # Pull 1cm backwards (closer to robot)

            try:
                # Pass both the Z and the R offset to the math
                new_angles = get_offset_angles(PLACE_1, z_offset, r_offset)
                stack_targets.append(new_angles)
                
                angles_str = ", ".join([f"{q:.4f}" for q in new_angles])
                self.get_logger().info(f"Calculated IK Block {i+1} (Z:+{z_offset}m, R:{r_offset}m): [{angles_str}]")
                
            except ValueError as e:
                self.get_logger().error(str(e))
                stack_targets.append(HOME)

        # Calculate Universal Clearances (Hovering 15cm above PLACE_1)
        try:
            hover_angles = get_offset_angles(PLACE_1, 0.150)
            UNIVERSAL_PRE_PLACE  = hover_angles[:5] + [GRIPPER_CLOSED]
            UNIVERSAL_POST_PLACE = hover_angles[:5] + [GRIPPER_OPEN]
        except ValueError:
            self.get_logger().warn("Hover height out of reach, using safe approximation.")
            UNIVERSAL_PRE_PLACE  = [PLACE_BASE, 0.0000, -0.8713, -1.3898, PLACE_TILT, GRIPPER_CLOSED]
            UNIVERSAL_POST_PLACE = [PLACE_BASE, 0.0000, -0.8713, -1.3898, PLACE_TILT, GRIPPER_OPEN]

        
        # ---------------------------------------------------------
        # OPTIMIZED Sequence Generation (~6.5s per block)
        # ---------------------------------------------------------
        self.sequence = [(HOME, 3.0, "Moving to Home")] # Skip the initial home to save 3 seconds right away!

        for i, target_place in enumerate(stack_targets):
            block_num = i + 1
            
            # 1. Calculate a Dynamic Hover (Only 4cm above THIS specific block's drop point)
            try:
                hover_angles = get_offset_angles(target_place, 0.040)
            except ValueError:
                hover_angles = get_offset_angles(target_place, 0.010) # Fallback to 1cm if max reach hit
                
            DYNAMIC_PRE_PLACE  = hover_angles[:5] + [GRIPPER_CLOSED]
            
            # 2. Overlap the Gripper opening with the retreat
            # By setting the gripper to OPEN here, the servo will actuate 
            # *during* the upward vertical movement.
            DYNAMIC_POST_PLACE = hover_angles[:5] + [GRIPPER_OPEN]

            self.sequence.extend([
                # --- PICK PHASE ---
                # Fast sweep to pick zone
                (PRE_GRASP,  1.3, f"[Block {block_num}] Sweeping to pick zone"),
                # Quick plunge
                (GRASP_0,    1.2, f"[Block {block_num}] Plunge"),
                # Fast bite and immediate lift
                (POST_GRASP, 0.9, f"[Block {block_num}] Bite & Lift"),
                
                # --- PLACE PHASE ---
                # Long lateral sweep to the dynamic hover point
                (DYNAMIC_PRE_PLACE,  1.7, f"[Block {block_num}] Sweeping to Stack"),
                # Gentle vertical drop to place the block (STABILITY FOCUS)
                (target_place,       1.7, f"[Block {block_num}] Gentle Drop"),
                # Fast vertical retreat while springing the gripper open
                (DYNAMIC_POST_PLACE, 1.0, f"[Block {block_num}] Fast Retreat & Release")
            ])
            
        # Optional: Return home only when the timer is likely up
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