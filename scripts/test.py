from hiwonder import HiwonderRobot
import time
import traceback

robot = HiwonderRobot()

def main():

    try:
        while True:
            print("\nControl Options:")
            print("0: Read individual joint position")
            print("1: Set motor speeds")
            print("2: Control individual joint")
            print("3: Set all joint positions")
            print("4: Exit")

            try:
                case = int(input("Select a control state (0-4): ").strip())
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 4.")
                continue

            if case == 0:
                try:
                    joint_id = int(input("Enter joint ID (1-6): ").strip())
                    print(f"Position of joint {joint_id}: {robot.joint_values[joint_id]}")
                except ValueError:
                    print("Invalid input. Please enter a valid joint ID.")

            elif case == 1:
                try:
                    speed = [int(s) for s in input("Enter speeds (e.g., 10,20,10,20): ").split(",")]
                    robot.board.set_motor_speed(speed)
                except ValueError:
                    print("Invalid input. Please enter comma-separated integers.")

            elif case == 2:
                try:
                    joint_id, theta = [int(s) for s in input("Enter joint ID and angle (e.g., 2,30): ").split(",")]
                    robot.set_joint_value(joint_id, theta)
                    time.sleep(1.0)
                    print(f"Position of joint {joint_id}: {robot.joint_values[joint_id]}")
                except ValueError:
                    print("Invalid input. Please enter valid numbers.")

            elif case == 3:
                try:
                    thetalist = [int(s) for s in input("Enter joint angles (e.g., 2,30,20,20,30,10): ").split(",")]
                    robot.set_joint_values(thetalist)
                    time.sleep(0.5)
                    print(f"Joint values: {robot.joint_values}")
                except ValueError:
                    print("Invalid input. Please enter valid numbers.")

            elif case == 4:
                print("Exiting program.")
                break

            else:
                print("Invalid selection. Please enter a number between 0 and 4.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping motors and exiting.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc() 

    finally:
        if robot is not None:
            shutdown_robot()  # Ensure a clean shutdown
        

def shutdown_robot():
    print("\n[INFO] Shutting down the robot safely...")

    # Stop all motors
    print("[INFO] Stopping all motors...")
    robot.stop_motors()

    # Move servos to a safe position
    print("[INFO] Resetting servo positions...")
    robot.set_joint_values([0, 0, 90, -30, 0, 0])

    time.sleep(1.5)  # Allow servos to reach their position

    # Close communication interfaces
    print("[INFO] Closing I2C and Serial connections...")
    robot.board.close()  # Close I2C bus
    robot.servo_bus.close()  # Ensure the serial port is properly closed

    print("\n[INFO] Shutdown complete. Safe to power off.")


if __name__ == "__main__":
    main()
