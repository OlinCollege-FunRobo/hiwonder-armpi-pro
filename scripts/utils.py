from typing import List
#from dataclasses import dataclass, field
import math, numpy as np
from math import sqrt, sin, cos, atan, atan2
from functools import singledispatch

PI = 3.1415926535897932384

#@dataclass
class State:
    """This dataclass represents the system state (pos and vel) """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    x_dot: float = 0.0
    y_dot: float = 0.0
    theta_dot: float = 0.0


#@dataclass
class Controls:
    """This dataclass represents the system controls """
    v: float = 0.0
    w: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


#@dataclass
class GamepadCmds:
    """This dataclass represents the gamepad commands """
    base_vx: int = 0
    base_vy: int = 0
    base_w: int = 0
    arm_vx: int = 0
    arm_vy: int = 0
    arm_vz: int = 0
    arm_j1: int = 0
    arm_j2: int = 0
    arm_j3: int = 0
    arm_j4: int = 0
    arm_j5: int = 0
    arm_ee: int = 0
    arm_home: int = 0
    utility_btn: int = 0

def print_dataclass(obj):
    print("------------------------------------")
    for field in obj.__dataclass_fields__:
        print(f"{field}: {round(getattr(obj, field), 3)}")
    print("------------------------------------ \n")

#@dataclass
class EndEffector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotx: float = 0.0
    roty: float = 0.0
    rotz: float = 0.0


def rotm_to_euler(R) -> tuple:
    """Converts a rotation matrix to Euler angles (roll, pitch, yaw).

    Args:
        R (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: Roll, pitch, and yaw angles (in radians).

    """
    
    r11 = R[0,0] if abs(R[0,0]) > 1e-7 else 0.0
    r12 = R[0,1] if abs(R[0,1]) > 1e-7 else 0.0
    r21 = R[1,0] if abs(R[1,0]) > 1e-7 else 0.0
    r22 = R[1,1] if abs(R[1,1]) > 1e-7 else 0.0
    r32 = R[2,1] if abs(R[2,1]) > 1e-7 else 0.0
    r33 = R[2,2] if abs(R[2,2]) > 1e-7 else 0.0
    r31 = R[2,0] if abs(R[2,0]) > 1e-7 else 0.0


    if abs(r31) != 1:
        roll = math.atan2(r32, r33)        
        yaw = math.atan2(r21, r11)
        denom = math.sqrt(r11 ** 2 + r21 ** 2)
        pitch = math.atan2(-r31, denom)
    
    elif r31 == 1:
        # pitch is close to -90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we choose one possible solution where yaw = 0
        # pitch, yaw = -PI/2, 0.0
        # roll = -math.atan2(r12, r22)
        pitch, roll = -PI/2, 0.0
        yaw = -math.atan2(r12, r22)
    
    elif r31 == -1:
        # pitch is close to 90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we choose one possible solution where yaw = 0
        # pitch, yaw = PI/2, 0.0
        # roll = math.atan2(r12, r22)
        pitch, roll = PI/2, 0.0
        yaw = math.atan2(r12, r22)
        

    return roll, pitch, yaw


def dh_to_matrix(dh_params: list) -> np.ndarray:
    """Converts Denavit-Hartenberg parameters to a transformation matrix.

    Args:
        dh_params (list): Denavit-Hartenberg parameters [theta, d, a, alpha].

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    theta, d, a, alpha = dh_params
    return np.array([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])


def euler_to_rotm(rpy: tuple) -> np.ndarray:
    """Converts Euler angles (roll, pitch, yaw) to a rotation matrix.

    Args:
        rpy (tuple): A tuple of Euler angles (roll, pitch, yaw).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                    [0, math.sin(rpy[0]), math.cos(rpy[0])]])
    R_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                    [0, 1, 0],
                    [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
    R_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                    [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


#@dataclass
##class SimData:
##    """Captures simulation data for storage.
##
##    Attributes:
##        x (List[float]): x-coordinates over time.
##        y (List[float]): y-coordinates over time.
##        theta (List[float]): Angles over time.
##        x_dot (List[float]): x-velocity over time.
##        y_dot (List[float]): y-velocity over time.
##        theta_dot (List[float]): Angular velocity over time.
##        v (List[float]): Linear velocity over time.
##        w (List[float]): Angular velocity over time.
##        vx (List[float]): x-component of linear velocity over time.
##        vy (List[float]): y-component of linear velocity over time.
##    """
##    x: List[float] = field(default_factory=list)
##    y: List[float] = field(default_factory=list)
##    theta: List[float] = field(default_factory=list)
##    x_dot: List[float] = field(default_factory=list)
##    y_dot: List[float] = field(default_factory=list)
##    theta_dot: List[float] = field(default_factory=list)
##    v: List[float] = field(default_factory=list)
##    w: List[float] = field(default_factory=list)
##    vx: List[float] = field(default_factory=list)
##    vy: List[float] = field(default_factory=list)


def check_joint_limits(theta: List[float], theta_limits: List[List[float]]) -> bool:
    """Checks if the joint angles are within the specified limits.

    Args:
        theta (List[float]): Current joint angles.
        theta_limits (List[List[float]]): Joint limits for each joint.

    Returns:
        bool: True if all joint angles are within limits, False otherwise.
    """
    for i, th in enumerate(theta):
        if not (theta_limits[i][0] <= th <= theta_limits[i][1]):
            return False
    return True


def calc_distance(p1: State, p2: State) -> float:
    """Calculates the Euclidean distance between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calc_heading(p1: State, p2: State) -> float:
    """Calculates the heading (angle) between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The heading angle in radians.
    """
    return atan2(p1.y - p2.y, p1.x - p2.x)


@singledispatch
def calc_angdiff(p1: State, p2: State) -> float:
    """Calculates the angular difference between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The angular difference in radians.
    """
    d = p1.theta - p2.theta
    return math.fmod(d, 2 * math.pi)


@calc_angdiff.register
def _(th1: float, th2: float) -> float:
    """Calculates the angular difference between two angles.

    Args:
        th1 (float): The first angle.
        th2 (float): The second angle.

    Returns:
        float: The angular difference in radians.
    """
    return math.fmod(th1 - th2, 2 * math.pi)


def near_zero(arr: np.ndarray) -> np.ndarray:
    """Checks if elements of an array are near zero.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        np.ndarray: An array with zeros where values are near zero, otherwise the original values.
    """
    tol = 1e-5
    return np.where(np.isclose(arr, 0, atol=tol), 0, arr)


def wraptopi(angle_rad):
  """Wraps an angle in radians to the range [-pi, pi).

  Args:
    angle_rad: The angle in radians.

  Returns:
    The wrapped angle in radians.
  """
  return (angle_rad + math.pi) % (2 * math.pi) - math.pi