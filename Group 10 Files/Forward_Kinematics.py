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

T_world_base  = H_xyzrpy([0, 0, 0], [0, 0, sp.pi])
T_base_shldr  = H_xyzrpy([0, S(-0.0452), S(0.0165)], [0, 0, 0]) * H_rot_z(q1)
T_shldr_upper = H_xyzrpy([0, S(-0.0306), S(0.1025)], [0, -sp.pi/2, 0]) * H_rot_z(q2)
T_upper_lower = H_xyzrpy([S(0.11257), S(-0.028), 0], [0, 0, 0]) * H_rot_z(q3)
T_lower_wrist = H_xyzrpy([S(0.0052), S(-0.1349), 0], [0, 0, sp.pi/2]) * H_rot_z(q4)
T_wrist_grip  = H_xyzrpy([S(-0.0601), 0, 0], [0, -sp.pi/2, 0]) * H_rot_z(q5)
T_grip_center = H_xyzrpy([0, 0, S(0.075)], [0, 0, 0])

T_fk = sp.trigsimp(
    T_world_base * T_base_shldr * T_shldr_upper *
    T_upper_lower * T_lower_wrist * T_wrist_grip * T_grip_center
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



