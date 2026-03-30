# =========================================================
# 9) FIND POSES WITH MULTIPLE IK SOLUTIONS
# =========================================================
def find_poses_with_multiple_solutions(bounds=None, tol=1e-2):
    poses = {
        "I":   [0.2, 0.2,    0.2,    0.000,  1.570,  0.650],
        "II":  [0.2, 0.1,    0.4,    0.000,  0.000, -1.570],
        "III": [0.0, 0.0,    0.4,    0.000, -0.785,  1.570],
        "IV":  [0.0, 0.0,    0.07,   3.141,  0.000,  0.000],
        "V":   [0.0, 0.0452, 0.45,  -0.785,  0.000,  3.141],
    }
    seeds = [
        np.array([0, 0, 0, 0, 0], dtype=float),
        np.array([ np.pi/2, 0, 0, 0, 0], dtype=float),
        np.array([-np.pi/2, 0, 0, 0, 0], dtype=float),
        np.array([0,  np.pi/4, -np.pi/4, 0, 0], dtype=float),
        np.array([0, -np.pi/4,  np.pi/4, 0, 0], dtype=float),
        np.array([0,  np.pi/2, 0, 0, 0], dtype=float),
        np.array([0, -np.pi/2, 0, 0, 0], dtype=float),
        np.array([np.pi, 0, 0, 0, 0], dtype=float),
        np.array([-np.pi, 0, 0, 0, 0], dtype=float),
    ]
    print("\n========== SEARCHING FOR MULTIPLE IK SOLUTIONS ==========")
    for name, pose in poses.items():
        solutions = []
        for seed in seeds:
            # Clip seed to bounds if bounds are provided
            clipped_seed = seed.copy()
            if bounds is not None:
                lower = np.array([b[0] for b in bounds], dtype=float)
                upper = np.array([b[1] for b in bounds], dtype=float)
                clipped_seed = np.clip(clipped_seed, lower, upper)
            try:
                q_sol, info, result = inverse_kinematics_pose5(
                    target_pose=pose,
                    initial_guess=clipped_seed,
                    bounds=bounds
                )
            except ValueError as e:
                # Skip infeasible seeds
                continue
            if info["feasible"]:
                # Check if this solution is distinct from previous ones
                is_new = True
                for q_prev in solutions:
                    # Use np.unwrap to handle angle wrapping
                    if np.linalg.norm(np.unwrap(q_sol - q_prev)) < tol:
                        is_new = False
                        break
                if is_new:
                    solutions.append(q_sol)
        if len(solutions) > 1:
            print(f"\nPose {name} has multiple ({len(solutions)}) solutions:")
            for idx, q in enumerate(solutions):
                print(f"  Solution {idx+1}: {np.round(q, 5)}")
        else:
            print(f"\nPose {name} has a unique solution.")

import sympy as sp
import numpy as np
from scipy.optimize import least_squares


# =========================================================
# 1) SYMBOLIC FK
# =========================================================
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


# Joint symbols
q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5', real=True)
q_syms = [q1, q2, q3, q4, q5]

# Robot FK chain
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

print("Position:")
sp.pprint(p_fk)

print("\nRotation:")
sp.pprint(R_fk)

test_cfg = {q1: 0, q2: 1.57, q3: 1.58, q4: 0, q5: 0}
print("\nNumerical test position:")
sp.pprint(p_fk.subs(test_cfg).evalf(6))


# =========================================================
# 2) LAMBDIFIED NUMERICAL FUNCTIONS
# =========================================================
fk_pos_func = sp.lambdify((q1, q2, q3, q4, q5), p_fk, "numpy")
fk_rot_func = sp.lambdify((q1, q2, q3, q4, q5), R_fk, "numpy")


# Use tool z-axis for the 2 orientation DoFs
z_tool_sym = R_fk[:, 2]

# 5D task = [x, y, z, z_tool_x, z_tool_y]
task_sym = sp.Matrix([
    p_fk[0],
    p_fk[1],
    p_fk[2],
    z_tool_sym[0],
    z_tool_sym[1]
])

task_func = sp.lambdify((q1, q2, q3, q4, q5), task_sym, "numpy")


# =========================================================
# 3) NUMERICAL ROTATION HELPERS
# =========================================================
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
    # Same convention as H_xyzrpy:
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return rotz_np(yaw) @ roty_np(pitch) @ rotx_np(roll)


# =========================================================
# 4) TASK CONVERSION
# =========================================================
def pose_to_task(target_pose):
    """
    target_pose = [x, y, z, roll, pitch, yaw]
    task       = [x, y, z, z_tool_x, z_tool_y]
    """
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


# =========================================================
# 5) SINGLE-START 5D NUMERICAL IK
# =========================================================
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
    """
    Solve IK for 3 position DoFs + 2 orientation DoFs.

    target_pose: [x, y, z, roll, pitch, yaw]
    initial_guess: shape (5,)
    bounds: [(q1_min, q1_max), ..., (q5_min, q5_max)]
    """

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


# =========================================================
# 6) MULTI-START WRAPPER
# =========================================================
def inverse_kinematics_pose5_multistart(
    target_pose,
    seeds=None,
    bounds=None,
    pos_weight=1.0,
    ori_weight=0.4,
    pos_tol=5e-3,
    ori_tol=5e-2
):
    if seeds is None:
        seeds = [
            np.array([0, 0, 0, 0, 0], dtype=float),
            np.array([ np.pi/2, 0, 0, 0, 0], dtype=float),
            np.array([-np.pi/2, 0, 0, 0, 0], dtype=float),
            np.array([0,  np.pi/4, -np.pi/4, 0, 0], dtype=float),
            np.array([0, -np.pi/4,  np.pi/4, 0, 0], dtype=float),
            np.array([0,  np.pi/2, 0, 0, 0], dtype=float),
            np.array([0, -np.pi/2, 0, 0, 0], dtype=float),
        ]

    best = None

    for seed in seeds:
        q_sol, info, result = inverse_kinematics_pose5(
            target_pose=target_pose,
            initial_guess=seed,
            bounds=bounds,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            pos_tol=pos_tol,
            ori_tol=ori_tol
        )

        score = info["pos_err"] + info["ori_err"]

        if (best is None) or (score < best["score"]):
            best = {
                "q": q_sol,
                "info": info,
                "result": result,
                "score": score,
                "seed": seed
            }

    return best["q"], best["info"], best["result"], best["seed"]


# =========================================================
# 7) TEST THE 5 ASSIGNMENT POSES
# =========================================================
def test_assignment_poses(bounds=None):
    poses = {
        "I":   [0.2, 0.2,    0.2,    0.000,  1.570,  0.650],
        "II":  [0.2, 0.1,    0.4,    0.000,  0.000, -1.570],
        "III": [0.0, 0.0,    0.4,    0.000, -0.785,  1.570],
        "IV":  [0.0, 0.0,    0.07,   3.141,  0.000,  0.000],
        "V":   [0.0, 0.0452, 0.45,  -0.785,  0.000,  3.141],  # 2nd "IV" in PDF
    }

    results = {}

    for name, pose in poses.items():
        q_sol, info, result, seed = inverse_kinematics_pose5_multistart(
            target_pose=pose,
            bounds=bounds
        )

        results[name] = {
            "target_pose": pose,
            "q_sol": q_sol,
            "used_seed": seed,
            "feasible": info["feasible"],
            "pos_err": info["pos_err"],
            "ori_err": info["ori_err"],
            "position": info["position"],
            "z_tool": info["z_tool"],
            "message": info["message"],
            "nfev": info["nfev"]
        }

    return results


def print_assignment_results(results):
    for name, data in results.items():
        print(f"\n--- Pose {name} ---")
        print("Target pose:", np.array(data["target_pose"]))
        print("Feasible   :", data["feasible"])
        print("q_sol [rad]:", np.round(data["q_sol"], 5))
        print("pos err [m]:", round(float(data["pos_err"]), 6))
        print("ori err    :", round(float(data["ori_err"]), 6))
        print("EE pos     :", np.round(data["position"], 5))
        print("tool z-axis:", np.round(data["z_tool"], 5))
        print("seed used  :", np.round(data["used_seed"], 5))
        print("nfev       :", data["nfev"])
        print("message    :", data["message"])


# =========================================================
# 8) MAIN
# =========================================================
if __name__ == "__main__":

    # Replace these with measured joint limits when you have them
    bounds = [
        (-2, 2),
        (-1.58, 1.58),
        (-1.58, 1.58),
        (-1.58, 1.58),
        (-np.pi, np.pi),
    ]

    # Single test
    pose_I = [0.2, 0.2, 0.2, 0.000, 1.570, 0.650]
    q_sol, info, result, seed = inverse_kinematics_pose5_multistart(
        target_pose=pose_I,
        bounds=bounds
    )

    print("\n========== SINGLE TEST: POSE I ==========")
    print("Best q [rad]:", np.round(q_sol, 5))
    print("Feasible    :", info["feasible"])
    print("Position err:", info["pos_err"])
    print("Orient. err :", info["ori_err"])
    print("Used seed   :", np.round(seed, 5))

    # Full assignment test
    results = test_assignment_poses(bounds=bounds)

    print("\n========== TASK 2.1 TEST RESULTS ==========")
    print_assignment_results(results)

    # Find and print poses with multiple solutions
    find_poses_with_multiple_solutions(bounds=bounds)
