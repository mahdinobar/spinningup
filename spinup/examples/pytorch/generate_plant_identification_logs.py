# ---------- make_prbs_u_log.py ----------
import numpy as np
from typing import Tuple
import pybullet as pb
import pybullet_data

def make_binary_prbs(A: float, dt: float, T: float, bit_T: float, seed: int = 0) -> np.ndarray:
    """
    Generate a simple binary PRBS: values in {+A, -A}, switching every 'bit_T'.
    Returns (N,) array where N = round(T/dt).
    """
    rng = np.random.default_rng(seed)
    N = int(np.round(T / dt))
    Nbit = max(1, int(np.round(bit_T / dt)))
    u = np.zeros(N, dtype=float)
    sign = 1.0
    for k in range(N):
        if k % Nbit == 0:
            sign = rng.choice([1.0, -1.0])
        u[k] = sign * A
    return u

def make_sequential_prbs_u_log(dt: float,
                               T_per_joint: float,
                               bit_T: float,
                               A_vec: np.ndarray,
                               seed0: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a u_log for 6 joints where we excite ONE joint at a time (others zero).
    For joint j: u_block[:, j] = PRBS(A_vec[j]), all other columns = 0.
    Then we stack the six blocks vertically to get the final (N_total, 6) u_log.

    Inputs
    ------
    dt : float
        Sampling time (s). For your setup: dt = 0.1
    T_per_joint : float
        Duration per joint (s); total duration = 6 * T_per_joint.
    bit_T : float
        PRBS bit duration (s); switching period.
    A_vec : (6,) array-like
        PRBS amplitude for each joint (rad/s).
    seed0 : int
        Base seed for random signs. Joint j uses seed (seed0 + j).

    Returns
    -------
    u_log : (N_total, 6) ndarray
        Commanded joint velocities to send to the robot/sim at 1/dt Hz.
    t     : (N_total,) ndarray
        Time stamps (s), t[0]=0, t[k]=k*dt.
    """
    A_vec = np.asarray(A_vec, float).reshape(6,)
    N_block = int(np.round(T_per_joint / dt))
    blocks = []
    for j in range(6):
        prbs_j = make_binary_prbs(A_vec[j], dt, T_per_joint, bit_T, seed=seed0 + j)  # (N_block,)
        block = np.zeros((N_block, 6), dtype=float)
        block[:, j] = prbs_j
        blocks.append(block)

    u_log = np.vstack(blocks)  # shape (6*N_block, 6)
    N_total = u_log.shape[0]
    t = np.arange(N_total, dtype=float) * dt
    return u_log, t


if __name__ == "__main__":
    # Your fixed sampling time:
    dt = 0.004 # seconds

    # Choose PRBS parameters:
    T_per_joint = 10.0   # seconds per joint
    bit_T = 0.2          # seconds per bit (switch rate)

    # Safe amplitudes (15% of your limits):
    vel_limits = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100])
    A_vec = 0.15 * vel_limits  # rad/s

    u_log, t = make_sequential_prbs_u_log(dt, T_per_joint, bit_T, A_vec, seed0=42)

    print("u_log shape:", u_log.shape)  # (6*N_block, 6)
    print("First 5 rows of u_log:\n", np.round(u_log[:5, :], 3))
    print("Total duration (s):", t[-1])

    physics_client = pb.connect(pb.DIRECT)
    dt_pb_sim = 1 / 240
    pb.setTimeStep(timeStep=dt_pb_sim, physicsClientId=physics_client)
    # # Set gravity
    pb.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    # Load URDFs
    # Load robot, target object and plane urdf
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())


    arm = pb.loadURDF("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/URDFs/fep3/panda_corrected_Nosc.urdf",
                      useFixedBase=True, physicsClientId=physics_client)
    pb.resetBasePositionAndOrientation(
        arm, [0, 0, 0], pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]), physicsClientId=physics_client)

    # Reset joint at initial angles
    q0=np.load("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/q_k0.npy")
    for i in range(6):
        pb.resetJointState(arm, i, q0[i], physicsClientId=physics_client)
    pb.resetJointState(arm, 7, 1.939142517407308, physicsClientId=physics_client)
    for j in [6] + list(range(8, 12)):
        pb.resetJointState(arm, j, 0, physicsClientId=physics_client)

    dq_log=[]
    for k in range(u_log.shape[0]):
        dqc_t = u_log[k,:]
        pb.setJointMotorControlArray(
            arm,
            [0, 1, 2, 3, 4, 5],
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocities=list(dqc_t),
            velocityGains=[1, 1, 2, 1, 1, 1],
            forces=[87, 87, 87, 87, 12, 12],
            physicsClientId=physics_client
        )
        # for _ in range(24):
        #     # default timestep is 1/240 second
        #     pb.stepSimulation(physicsClientId=physics_client)
        pb.stepSimulation(physicsClientId=physics_client)
        info = pb.getJointStates(arm, range(12), physicsClientId=physics_client)
        dq_measured = []
        for joint_info in info:
            dq_measured.append(joint_info[1])
        dq_measured = np.array(dq_measured)[:6]
        dq_log.append(dq_measured)


    np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/u_log_cor.npy",u_log)
    np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/t_cor.npy",t)
    np.save("/home/mahdi/ETHZ/codes/spinningup/spinup/examples/pytorch/logs/Fep_HW_314/dq_log_cor.npy",np.asarray(dq_log))
    print("")


