# hiwonder.py
"""
Hiwonder Robot Controller
-------------------------
Handles the control of the mobile base and 5-DOF robotic arm using commands received from the gamepad.
"""

import time
import numpy as np
from numpy import round
from board_controller import BoardController
from servo_bus_controller import ServoBusController
import utils as ut
import sympy as sp
import math


# Robot base constants
WHEEL_RADIUS = 0.047  # meters
BASE_LENGTH_X = 0.096  # meters
BASE_LENGTH_Y = 0.105  # meters

class HiwonderRobot:
    def __init__(self):
        """Initialize motor controllers, servo bus, and default robot states."""
        self.board = BoardController()
        self.servo_bus = ServoBusController()

        # in meters
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.155, 0.099, 0.095, 0.055, 0.105
        
        # should be radians
        # self.theta = [0, 0, 0, 0, 0]

        
        self.joint_values = [0, 0, 90, -30, 0, 0]  # degrees
        self.home_position = [0, 0, 90, -30, 0, 0]  # degrees
        
        self.joint_limits = [
             [-120, 120], [-90, 90], [-120, 120],
             [-100, 100], [-90, 90], [-120, 30]
        ]
        # self.theta_limits = [
        #     [-np.pi, np.pi], 
        #     [-np.pi/3, np.pi], 
        #     [-np.pi+np.pi/12, np.pi-np.pi/4], 
        #     [-np.pi+np.pi/12, np.pi-np.pi/12], 
        #     [-np.pi, np.pi]
        # ]
        self.joint_control_delay = 0.2 # secs
        self.speed_control_delay = 0.2

        self.move_to_home_position()

    # -------------------------------------------------------------
    # Methods for interfacing with the mobile base
    # -------------------------------------------------------------

    def set_robot_commands(self, cmd: ut.GamepadCmds):
        """Updates robot base and arm based on gamepad commands.

        Args:
            cmd (GamepadCmds): Command data class with velocities and joint commands.
        """

        if cmd.arm_home:
            self.move_to_home_position()

        print(f'---------------------------------------------------------------------')
        
        # self.set_base_velocity(cmd)
        self.set_arm_velocity(cmd)

        ######################################################################

        position = [0]*3
        
        ######################################################################

        print(f'[DEBUG] XYZ position: X: {round(position[0], 3)}, Y: {round(position[1], 3)}, Z: {round(position[2], 3)} \n')


    def set_base_velocity(self, cmd: ut.GamepadCmds): # Driving the robot, N/A
        """ Computes wheel speeds based on joystick input and sends them to the board """
        """
        motor3 w0|  ↑  |w1 motor1
                 |     |
        motor4 w2|     |w3 motor2
        
        """
        ######################################################################
        # insert your code for finding "speed"

        speed = [0]*4
        
        ######################################################################

        # Send speeds to motors
        self.board.set_motor_speed(speed)
        time.sleep(self.speed_control_delay)

    # -------------------------------------------------------------
    # Methods for interfacing with the 5-DOF robotic arm
    # -------------------------------------------------------------

    def set_arm_velocity(self, cmd: ut.GamepadCmds):
        """Calculates and sets new joint angles from linear velocities.

        Args:
            cmd (GamepadCmds): Contains linear velocities for the arm.
        """
        vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]
        ######################################################################
        # insert your code for finding "thetalist_dot" (FVK)
        
        t0, t1, t2, t3, t4  = sp.symbols('t0 t1 t2 t3 t4 ')

        arr = []
        dhTable = [[t0, self.l1, 0, math.pi/2],
                   [math.pi/2, 0, 0, 0],
                   [t1, 0, self.l2, math.pi],
                   [t2, 0, self.l3, math.pi], 
                   [t3, 0, self.l4, 0],
                   [-math.pi/2, 0, 0, -math.pi/2],
                   [t4, self.l5, 0, 0]]
        
        for i in range(len(dhTable)):
            arr.append(ut.dh_sympi_to_matrix(dhTable[i]))
        
        #print(arr[0])
        #print(type(arr[0]))
        Hm = (arr[0] * (arr[1] * (arr[2] * (arr[3] * (arr[4] * (arr[5] * arr[6]))))))
        #print(arr)
        


        #Hm = m01j * m12j * m23j * m34j * m45j * m56j
        #print(Hm)
        Hx = Hm[0, 3]
        Hy = Hm[1 , 3]
        Hz = Hm[2 , 3]

        # [row, column]
    
        jacobian = sp.Matrix([[sp.diff(Hx, t0), sp.diff(Hx, t1), sp.diff(Hx, t2),sp.diff(Hx, t3), sp.diff(Hx, t4)],
                           [sp.diff(Hy, t0), sp.diff(Hy, t1), sp.diff(Hy, t2),sp.diff(Hy, t3), sp.diff(Hy, t4)],
                           [sp.diff(Hz, t0), sp.diff(Hz, t1), sp.diff(Hz, t2),sp.diff(Hz, t3), sp.diff(Hz, t4)],
                           ])


        # This will not work because of variable names
        # print(self.joint_values)
        jacobian = jacobian.evalf(subs={t0: sp.rad(self.joint_values[0])})
        jacobian =  jacobian.evalf(subs={t1: sp.rad(self.joint_values[1])})
        jacobian =  jacobian.evalf(subs={t2: sp.rad(self.joint_values[2])})
        jacobian =  jacobian.evalf(subs={t3: sp.rad(self.joint_values[3])})
        jacobian =  jacobian.evalf(subs={t4: sp.rad(self.joint_values[4])})

        #print("Jacobian", jacobian)
        #print()

        invJac =  np.array(sp.transpose(jacobian) * (( (jacobian * sp.transpose(jacobian)) + sp.eye(3)*.0001) **-1 ))
       # print( jacobian * sp.transpose(jacobian)* sp.eye(3)*1.0001) 
        #print("hi", invJac)
        npVel = np.array([vel])

        max_vel = 2

        for i in range(3):
            if npVel[0][i]> max_vel:
                npVel[0][i] = max_vel
            elif npVel[0][i] < -max_vel:
                npVel[0][i] = -max_vel
        #print("shape of vel", np.shape(npVel))
        #print("shape of invJac",  np.shape(invJac))
        #print(invJac)
        thetaDot = np.matmul(invJac, np.transpose(npVel))
        #print("thetadot shape", np.shape(thetaDot))

        # print(cmd.arm_j1)
        ######################################################################


        print(f'[DEBUG] Current thetalist (deg) = {self.joint_values}') 
        print(f'[DEBUG] linear vel: {npVel=}')
        print(f'[DEBUG] thetadot (deg/s) = {thetaDot=}')

        # Update joint angles
        dt = 0.2 # Fixed time step
        K_ind = 200 # mapping gain for individual joint control
        #K_tot = 1
        new_thetalist = [0.0]*6

        # linear velocity control
        for i in range(5):
            new_thetalist[i] = self.joint_values[i] + dt * np.rad2deg(float((thetaDot[i][0]))) # thetalist_dot[i]
       
        # individual joint control
        new_thetalist[0] += dt * K_ind * cmd.arm_j1
        new_thetalist[1] += dt * K_ind * cmd.arm_j2
        new_thetalist[2] += dt * K_ind * cmd.arm_j3
        new_thetalist[3] += dt * K_ind * cmd.arm_j4
        new_thetalist[4] += dt * K_ind * cmd.arm_j5
        new_thetalist[5] = self.joint_values[5] + dt * K_ind * cmd.arm_ee

        new_thetalist = [round(theta,2) for theta in new_thetalist]
        print(f'[DEBUG] Commanded thetalist (deg) = {new_thetalist}')       
        
        # set new joint angles
        self.set_joint_values(new_thetalist, radians=False)


    def set_joint_value(self, joint_id: int, theta: float, duration=250, radians=False):
        """ Moves a single joint to a specified angle """
        if not (1 <= joint_id <= 6):
            raise ValueError("Joint ID must be between 1 and 6.")

        if radians:
            theta = np.rad2deg(theta)

        theta = self.enforce_joint_limits(theta, joint_id=joint_id)
        self.joint_values[joint_id] = theta

        pulse = self.angle_to_pulse(theta)
        self.servo_bus.move_servo(joint_id, pulse, duration)
        
        print(f"[DEBUG] Moving joint {joint_id} to {theta}° ({pulse} pulse)")
        time.sleep(self.joint_control_delay)


    def set_joint_values(self, thetalist: list, duration=250, radians=False):
        """Moves all arm joints to the given angles.

        Args:
            thetalist (list): Target joint angles in degrees.
            duration (int): Movement duration in milliseconds.
        """
        if len(thetalist) != 6:
            raise ValueError("Provide 6 joint angles.")

        if radians:
            thetalist = [np.rad2deg(theta) for theta in thetalist]

        thetalist = self.enforce_joint_limits(thetalist)
        self.joint_values = thetalist # updates joint_values with commanded thetalist
        thetalist = self.remap_joints(thetalist) # remap the joint values from software to hardware

        for joint_id, theta in enumerate(thetalist, start=1):
            pulse = self.angle_to_pulse(theta)
            self.servo_bus.move_servo(joint_id, pulse, duration)


    def enforce_joint_limits(self, thetalist: list) -> list:
        """Clamps joint angles within their hardware limits.

        Args:
            thetalist (list): List of target angles.

        Returns:
            list: Joint angles within allowable ranges.
        """
        return [np.clip(theta, *limit) for theta, limit in zip(thetalist, self.joint_limits)]


    def move_to_home_position(self):
        print(f'Moving to home position...')
        self.set_joint_values(self.home_position, duration=800)
        time.sleep(2.0)
        print(f'Arrived at home position: {self.joint_values} \n')
        time.sleep(1.0)
        print(f'------------------- System is now ready!------------------- \n')


    # -------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------

    def angle_to_pulse(self, x: float):
        """ Converts degrees to servo pulse value """
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return int((x - joint_min) * (hw_max - hw_min) / (joint_max - joint_min) + hw_min)


    def pulse_to_angle(self, x: float):
        """ Converts servo pulse value to degrees """
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return round((x - hw_min) * (joint_max - joint_min) / (hw_max - hw_min) + joint_min, 2)


    def stop_motors(self):
        """ Stops all motors safely """
        self.board.set_motor_speed([0]*4)
        print("[INFO] Motors stopped.")


    def remap_joints(self, thetalist: list):
        """Reorders angles to match hardware configuration.

        Args:
            thetalist (list): Software joint order.

        Returns:
            list: Hardware-mapped joint angles.

        Note: Joint mapping for hardware
            HARDWARE - SOFTWARE
            joint[0] = gripper/EE
            joint[1] = joint[5] 
            joint[2] = joint[4] 
            joint[3] = joint[3] 
            joint[4] = joint[2] 
            joint[5] = joint[1] 
        """
        return thetalist[::-1]