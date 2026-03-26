#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class PickAndPlaceEduBot(Node):

    def __init__(self):
        super().__init__('pick_and_place_edubot')

        # Publish directly to the topic, just like ExampleTraj
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

# ---------------------------------------------------------
        # Waypoints Configuration: 5-Block Stack
        # ---------------------------------------------------------
        GRIPPER_OPEN = 0.6
        GRIPPER_CLOSED = 0.3

        HOME       = [ 0.1733,  1.1720, -1.1520, -1.1950, -1.5355, GRIPPER_OPEN]
        
        # --- THE PICK COLUMN (Locked Base at -0.1795, Tilt at -1.5300) ---
        PRE_GRASP  = [-0.1795, -0.0460, -0.4556, -1.1244, -1.5300, GRIPPER_OPEN]
        GRASP_0    = [-0.1795, -0.4571, -0.2807, -1.1183, -1.5300, GRIPPER_CLOSED]
        POST_GRASP = [-0.1795, -0.0460, -0.4556, -1.1244, -1.5300, GRIPPER_CLOSED]

        # --- THE PLACE COLUMN (Locked Base at 0.9100, Tilt at -0.7480) ---
        PLACE_BASE = 0.9100
        PLACE_TILT = -0.7480
        
        # The 5 stacking heights (only joints 2, 3, 4 change)
        PLACE_1 = [PLACE_BASE, 0.2715, -1.1137, -1.0324, PLACE_TILT, GRIPPER_CLOSED]
        PLACE_2 = [PLACE_BASE, 0.2163, -0.8483, -1.3576, PLACE_TILT, GRIPPER_CLOSED]
        PLACE_3 = [PLACE_BASE, 0.2777, -0.7286, -1.4788, PLACE_TILT, GRIPPER_CLOSED]
        PLACE_4 = [PLACE_BASE, 0.3145, -0.6289, -1.6015, PLACE_TILT, GRIPPER_CLOSED]
        PLACE_5 = [PLACE_BASE, 0.2531, -0.4218, -1.7671, PLACE_TILT, GRIPPER_CLOSED]
        
        # List of our drop-off targets in order
        stack_targets = [PLACE_1, PLACE_2, PLACE_3, PLACE_4, PLACE_5]

        # Universal Safe Clearance (Hovers high above the stack)
        # Using Upper Arm at 0.0 to ensure it clears Block 5
        UNIVERSAL_PRE_PLACE  = [PLACE_BASE, 0.0000, -0.8713, -1.3898, PLACE_TILT, GRIPPER_CLOSED]
        UNIVERSAL_POST_PLACE = [PLACE_BASE, 0.0000, -0.8713, -1.3898, PLACE_TILT, GRIPPER_OPEN]

        # ---------------------------------------------------------
        # Programmatic Sequence Generation
        # ---------------------------------------------------------
        self.sequence = [
            (HOME, 3.0, "Moving to Home")
        ]

        # Loop 5 times to generate the steps for all 5 blocks dynamically
        for i, target_place in enumerate(stack_targets):
            block_num = i + 1
            self.sequence.extend([
                # Pick Phase
                (PRE_GRASP,  2.5, f"[Block {block_num}] Moving above pick zone"),
                (GRASP_0,    2.0, f"[Block {block_num}] Lowering to pick object"),
                (POST_GRASP, 1.5, f"[Block {block_num}] Closing Gripper & Lifting"),
                
                # Place Phase
                (UNIVERSAL_PRE_PLACE, 3.5, f"[Block {block_num}] Moving high above drop zone"),
                (target_place,        3.5, f"[Block {block_num}] Lowering to Stack Level {block_num}"),
                (UNIVERSAL_POST_PLACE,3.5, f"[Block {block_num}] Opening Gripper & Lifting safely")
            ])

        # Finally, return home after all 5 are stacked
        self.sequence.append((HOME, 3.0, "Sequence Complete! Returning Home"))

        # State tracking
        self.current_step = 0
        self.start_config = HOME  # Assume robot starts at or near HOME
        self.start_time = self.get_clock().now()

        # Run the timer at 50 Hz (0.02s) to ensure smooth streaming
        self._timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info(f"Starting Sequence: {self.sequence[0][2]}")

    def smooth_step(self, x):
        """Easing function so the robot accelerates and decelerates smoothly."""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def timer_callback(self):
        if self.current_step >= len(self.sequence):
            return  # Sequence finished

        now = self.get_clock().now()
        target_config, duration, label = self.sequence[self.current_step]
        
        # Calculate how much time has passed in the current step
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        # If time is up, move to the next step
        if elapsed >= duration:
            self.get_logger().info(f"Completed: {label}")
            self.current_step += 1
            
            if self.current_step >= len(self.sequence):
                self.get_logger().info("=== Pick and Place Complete! ===")
                self._timer.cancel()
                return
            
            # Setup the next movement
            self.start_config = target_config
            self.start_time = now
            next_label = self.sequence[self.current_step][2]
            self.get_logger().info(f"Starting Sequence: {next_label}")
            return

        # Interpolate between the start and target configurations
        alpha = elapsed / duration
        alpha = self.smooth_step(alpha)

        current_positions = []
        for i in range(6):
            start_val = self.start_config[i]
            target_val = target_config[i]
            current_positions.append(start_val + alpha * (target_val - start_val))

        # Build and publish the message
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