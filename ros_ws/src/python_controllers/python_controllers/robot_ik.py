"""
Inverse Kinematics for a 5-DOF Robot Arm
Uses robotics-toolbox-python (https://petercorke.github.io/robotics-toolbox-python/IK/ik.html)

Robot structure (from URDF):
  world -> base         : T=[0,0,0],        R=[0, 0, 3.14159]
  base -> shoulder      : T=[0,-0.0452,0.0165], R=[0,0,0]
  shoulder -> upper_arm : T=[0,-0.0306,0.1025], R=[0,-1.57079,0]
  upper_arm -> lower_arm: T=[0.11257,-0.028,0],  R=[0,0,0]
  lower_arm -> wrist    : T=[0.0052,-0.1349,0],  R=[0,0,1.57079]
  wrist -> gripper      : T=[-0.0601,0,0],       R=[0,-1.57079,0]

All 5 joints rotate about their local Z axis.

Target poses (x, y, z; roll, pitch, yaw) from image 2:
  I.   [0.2, 0.2, 0.2;  0.000,  1.570,  0.650]
  II.  [0.2, 0.1, 0.4;  0.000,  0.000, -1.570]
  III. [0.0, 0.0, 0.4;  0.000, -0.785,  1.570]
  IV.  [0.0, 0.0, 0.07; 3.141,  0.000,  0.000]
  IV.  [0.0, 0.0452, 0.45; -0.785, 0.000, 3.141]   (second IV / V)
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build the robot using DHRobot / ERobot from URDF transforms
#    We model each joint as a revolute joint rotating about Z.
#    The fixed transforms between joints are captured as ETS elements.
# ─────────────────────────────────────────────────────────────────────────────

def build_robot():
    """
    Construct the 5-DOF robot arm using the ETS (Elementary Transform Sequence)
    approach so we can capture arbitrary fixed translations + rotations between
    joints exactly as specified in the URDF table.
    """
    from roboticstoolbox.robot.ETS import ETS
    import roboticstoolbox as rtb

    # Helper: fixed SE3 from translation + RPY
    def fixed(xyz, rpy):
        return SE3(xyz[0], xyz[1], xyz[2]) * SE3.RPY(rpy, order='xyz')

    # ── Joint 1: base → shoulder (revolute Z) ──────────────────────────────
    # world→base fixed offset first
    T_world_base   = fixed([0, 0, 0],              [0, 0, np.pi])
    T_base_shoulder = fixed([0, -0.0452, 0.0165], [0, 0, 0])

    # ── Joint 2: shoulder → upper_arm ─────────────────────────────────────
    T_shoulder_ua  = fixed([0, -0.0306, 0.1025],  [0, -np.pi/2, 0])

    # ── Joint 3: upper_arm → lower_arm ────────────────────────────────────
    T_ua_la        = fixed([0.11257, -0.028, 0],   [0, 0, 0])

    # ── Joint 4: lower_arm → wrist ────────────────────────────────────────
    T_la_wrist     = fixed([0.0052, -0.1349, 0],   [0, 0, np.pi/2])

    # ── Joint 5: wrist → gripper ──────────────────────────────────────────
    T_wrist_gripper = fixed([-0.0601, 0, 0],       [0, -np.pi/2, 0])

    # ── End-effector: gripper → gripper_center ────────────────────────────
    T_gripper_ee   = fixed([0, 0, 0.075],           [0, 0, 0])

    # Build links using ERobot with ETS
    links = [
        rtb.Link(
            rtb.ETS(rtb.ET.SE3(T_world_base.A)) *
            rtb.ETS(rtb.ET.SE3(T_base_shoulder.A)) *
            rtb.ET.Rz(),
            name="joint1"
        ),
        rtb.Link(
            rtb.ETS(rtb.ET.SE3(T_shoulder_ua.A)) *
            rtb.ET.Rz(),
            name="joint2",
            parent="joint1"
        ),
        rtb.Link(
            rtb.ETS(rtb.ET.SE3(T_ua_la.A)) *
            rtb.ET.Rz(),
            name="joint3",
            parent="joint2"
        ),
        rtb.Link(
            rtb.ETS(rtb.ET.SE3(T_la_wrist.A)) *
            rtb.ET.Rz(),
            name="joint4",
            parent="joint3"
        ),
        rtb.Link(
            rtb.ETS(rtb.ET.SE3(T_wrist_gripper.A)) *
            rtb.ET.Rz(),
            name="joint5",
            parent="joint4"
        ),
    ]

    robot = rtb.ERobot(links, name="5DOF_Arm", tool=T_gripper_ee)
    return robot


# ─────────────────────────────────────────────────────────────────────────────
# 2. Target poses
# ─────────────────────────────────────────────────────────────────────────────

targets_raw = [
    ("I",   [0.2,     0.2,    0.2   ],  [0.000,  1.570,  0.650]),
    ("II",  [0.2,     0.1,    0.4   ],  [0.000,  0.000, -1.570]),
    ("III", [0.0,     0.0,    0.4   ],  [0.000, -0.785,  1.570]),
    ("IV",  [0.0,     0.0,    0.07  ],  [3.141,  0.000,  0.000]),
    ("V",   [0.0,     0.0452, 0.45  ],  [-0.785, 0.000,  3.141]),
]

def make_pose(xyz, rpy):
    return SE3(xyz[0], xyz[1], xyz[2]) * SE3.RPY(rpy, order='xyz')


# ─────────────────────────────────────────────────────────────────────────────
# 3. IK solver – try multiple seeds to find all distinct solutions
# ─────────────────────────────────────────────────────────────────────────────

N_SEEDS   = 200          # random seeds per target
TOL       = 1e-4         # position tolerance to accept a solution
ANGLE_TOL = np.deg2rad(5) # cluster threshold: two solutions are "different" if
                           # any joint differs by more than this

def solutions_are_different(q1, q2, tol=ANGLE_TOL):
    return np.any(np.abs(np.array(q1) - np.array(q2)) > tol)

def find_all_ik(robot, Tep, n_seeds=N_SEEDS, method="LM"):
    """
    Run IK from many random seeds and collect distinct solutions.
    Returns list of (q, residual_error) tuples.
    """
    distinct = []

    for seed_idx in range(n_seeds):
        # Random initial joint config within ±π
        q0 = np.random.uniform(-np.pi, np.pi, robot.n)

        if method == "LM":
            sol = robot.ik_LM(Tep, q0=q0, tol=1e-6, ilimit=1000, slimit=1)
        elif method == "GN":
            sol = robot.ik_GN(Tep, q0=q0, tol=1e-6, ilimit=1000, slimit=1)
        else:
            sol = robot.ik_NR(Tep, q0=q0, tol=1e-6, ilimit=1000, slimit=1)

        if not sol.success:
            continue

        q_sol = sol.q

        # Check actual end-effector error
        T_actual = robot.fkine(q_sol)
        pos_err  = np.linalg.norm(T_actual.t - Tep.t)
        if pos_err > TOL:
            continue

        # Normalise angles to [-π, π]
        q_norm = (q_sol + np.pi) % (2 * np.pi) - np.pi

        # Check if this solution is new
        is_new = all(solutions_are_different(q_norm, q_prev)
                     for q_prev, _ in distinct)
        if is_new:
            distinct.append((q_norm, pos_err))

    return distinct


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("Building robot…")
    robot = build_robot()
    print(robot)
    print(f"\nDOF: {robot.n}")
    print("=" * 70)

    for label, xyz, rpy in targets_raw:
        Tep = make_pose(xyz, rpy)

        print(f"\n{'='*70}")
        print(f"Point {label}:  xyz={xyz}   rpy={rpy}")
        print(f"Target SE3:\n{Tep}")

        solutions = find_all_ik(robot, Tep)

        if not solutions:
            print("  ✗ No IK solution found for this pose.")
        else:
            print(f"  ✓ Found {len(solutions)} distinct solution(s):\n")
            for i, (q, err) in enumerate(solutions, 1):
                q_deg = np.degrees(q)
                print(f"  Solution {i}  (pos_error={err:.2e} m)")
                print(f"    Joint angles [rad]: {np.round(q, 4)}")
                print(f"    Joint angles [deg]: {np.round(q_deg, 2)}")
                # Verify with FK
                T_fk = robot.fkine(q)
                print(f"    FK position:   {np.round(T_fk.t, 4)}")
                print(f"    Target position: {xyz}")
                print()

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()