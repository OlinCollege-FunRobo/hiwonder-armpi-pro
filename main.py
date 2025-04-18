# main.py
"""
Main Application Script
----------------------------
Coordinates gamepad input and robot control.
"""

import sys, os
import time
import threading
import traceback

# Extend system path to include script directory
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

from hiwonder import HiwonderRobot
from gamepad_control import GamepadControl
import utils


# Initialize components
cmdlist = []    # Stores recent gamepad commands
gpc = GamepadControl()
robot = HiwonderRobot()


def monitor_gamepad():
    """ Continuously reads gamepad inputs and stores the latest command. """
    try:
        while True:
            if len(cmdlist) > 2:
                cmdlist.pop(0)  # Retain only the latest two commands
            cmdlist.append(gpc.get_gamepad_cmds())
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("[INFO] Gamepad monitoring stopped.")


def shutdown_robot():
    print("\n[INFO] Shutting down the robot safely...")

    # Stop motors and reset servos to a safe position
    # robot.stop_motors()
    robot.set_joint_values(robot.home_position, duration=600)
    time.sleep(1.5)  # Allow time for servos to reposition

    # Close communication interfaces
    print("[INFO] Closing hardware interfaces...")
    # robot.board.close()
    # robot.servo_bus.close()

    print("[INFO] Shutdown complete. Safe to power off.")


def main():
    """ Main loop that reads gamepad commands and updates the robot accordingly. """
    try:
        # Start the gamepad monitoring thread
        gamepad_thread = threading.Thread(target=monitor_gamepad, daemon=True)
        gamepad_thread.start()
        
        control_interval = 0.25  # Seconds per control cycle
        
        while True:
            cycle_start = time.time()

            if cmdlist:
                latest_cmd = cmdlist[-1]
                # accessing the utility button, LB, on the gamepad (uncomment code below to use)
                # print(f'Utility Button is [ {latest_cmd.utility_btn} ]')
                
                robot.set_robot_commands(latest_cmd)

            elapsed = time.time() - cycle_start
            remaining_time = control_interval - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)
            
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Initiating shutdown...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        shutdown_robot()




if __name__ == "__main__":
    main()


