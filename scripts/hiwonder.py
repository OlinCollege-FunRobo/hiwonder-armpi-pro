# hiwonder.py
"""
Hiwonder Robot Controller
-------------------------
Handles the control of the mobile base and 5-DOF robotic arm using commands received from the gamepad.
"""

import time
import numpy as np
#from ros_robot_controller_sdk import Board
#from bus_servo_control import *
from ServoCmd import *

import utils as ut

# Robot base constants
WHEEL_RADIUS = 0.047  # meters
BASE_LENGTH_X = 0.096  # meters
BASE_LENGTH_Y = 0.105  # meters

class HiwonderRobot:
    def __init__(self):
        """Initialize motor controllers, servo bus, and default robot states."""

        self.joint_values = [0, 0, 90, -30, 0, 0]  # degrees
        self.home_position = [0, 0, 90, -30, 0, 0]  # degrees
        self.joint_limits = [
            [-120, 120], [-90, 90], [-120, 120],
            [-100, 100], [-90, 90], [-120, 30]
        ]
        self.joint_control_delay = 0.2 # secs
        self.speed_control_delay = 0.2
        self.time_out = 100

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
        
        # update joint values
        self.update_joint_values()

        print(f'Joint values: {self.get_joint_values()}')

        print(f'[DEBUG] XYZ position: X: {round(position[0], 3)}, Y: {round(position[1], 3)}, Z: {round(position[2], 3)} \n')


    def set_base_velocity(self, cmd: ut.GamepadCmds):
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
        #self.board.set_motor_speed(speed)
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
        # insert your code for finding "thetalist_dot"

        thetalist_dot = [0]*5

        ######################################################################


        print(f'[DEBUG] Current thetalist (deg) = {self.joint_values}') 
        print(f'[DEBUG] linear vel: {[round(vel[0], 3), round(vel[1], 3), round(vel[2], 3)]}')
        print(f'[DEBUG] thetadot (deg/s) = {[round(td,2) for td in thetalist_dot]}')

        # Update joint angles
        dt = 0.5 # Fixed time step
        K = 1600 # mapping gain for individual joint control
        new_thetalist = [0.0]*6

        # linear velocity control
        for i in range(5):
            new_thetalist[i] = self.joint_values[i] + dt * thetalist_dot[i]
        # individual joint control
        new_thetalist[0] += dt * K * cmd.arm_j1
        new_thetalist[1] += dt * K * cmd.arm_j2
        new_thetalist[2] += dt * K * cmd.arm_j3
        new_thetalist[3] += dt * K * cmd.arm_j4
        new_thetalist[4] += dt * K * cmd.arm_j5
        new_thetalist[5] = self.joint_values[5] + dt * K * cmd.arm_ee

        new_thetalist = [round(theta,2) for theta in new_thetalist]
        print(f'[DEBUG] Commanded thetalist (deg) = {new_thetalist}')       
        
        # set new joint angles
        self.set_joint_values(new_thetalist, radians=False)


    def set_joint_value(self, joint_id: int, theta: float, duration=250, radians=False):
        """ Moves a single joint to a specified angle """
        if not (0 <= joint_id <= 5):
            raise ValueError("Joint ID must be between 0 and 5.")

        if radians:
            theta = np.rad2deg(theta)

        #theta = self.enforce_joint_limits(theta)
        pulse = self.angle_to_pulse(theta)
        setServoPulse(joint_id, pulse, 800)
        
        print(f"[DEBUG] Moving joint {joint_id} to {theta}° ({pulse} pulse)")
        time.sleep(self.joint_control_delay)
        
        # update the joint value
        self.update_joint_value(joint_id)
        


    def set_joint_values(self, thetalist: list, duration=1000, radians=False):
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
        #self.joint_values = thetalist # updates joint_values with commanded thetalist
        thetalist = self.remap_joints(thetalist) # remap the joint values from software to hardware
        
        #positions = []
        for joint_id, theta in enumerate(thetalist, start=1):
            pulse = self.angle_to_pulse(theta)
            #positions.append([joint_id, pulse])
            setServoPulse(joint_id, pulse, 800)
        #self.board.bus_servo_set_position(1, positions)
        time.sleep(self.joint_control_delay)
        
        # update the joint values
        self.update_joint_values()
    
    
    def update_joint_value(self, joint_id: int):
        """ Gets the joint angle """
        if not (0 <= joint_id <= 5):
            raise ValueError("Joint ID must be between 0 and 5.")
        count = 0 
        while True:
            #print(f'Finding joint {joint_id}...')
            #res = self.board.bus_servo_read_position(joint_id)
            res = getServoPulse(joint_id+1)
            count += 1
            #print(f'Finding joint {joint_id} data: {res}')
            if res is not None:
                return res
            if count > self.time_out:
                print(f'Joint update for {joint_id} is timing out!')
                return None
            time.sleep(0.05)
            
    
    def update_joint_values(self):
        """ Updates the joint angle values by calling "get_joint_value" for all joints """
        res = [self.update_joint_value(i) for i in range(len(self.joint_values))]
        res = self.remap_joints(res)
        
        # check for Nones and replace with the previous value
        for i in range(len(res)):
            if res[i] is None:
                res[i] = self.joint_values[i]
            else:
                res[i] = self.pulse_to_angle(res[i])
        self.joint_values = res
        
    
    def get_joint_value(self, joint_id: int):
        """ Gets the joint angle """
        if not (0 <= joint_id <= 5):
            raise ValueError("Joint ID must be between 0 and 5.")
        return self.joint_values[joint_id]
    
    
    def get_joint_values(self):
        """ Returns all the joint angle values """
        return self.joint_values
    

    def enforce_joint_limits(self, thetalist: list) -> list:
        """Clamps joint angles within their hardware limits.

        Args:
            thetalist (list): List of target angles.

        Returns:
            list: Joint angles within allowable ranges.
        """
        return [np.clip(theta, *limit) for theta, limit in zip(thetalist, self.joint_limits)]


    def move_to_home_position(self):
        #self.board.set_buzzer(2400, 0.1, 0.9, 1)
        time.sleep(2)
    
        print(f'Moving to home position...')
        self.set_joint_values(self.home_position, duration=1000)
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