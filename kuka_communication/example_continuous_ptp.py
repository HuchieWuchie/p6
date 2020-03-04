import sys
sys.path.insert(0, 'include/')

from openshowvar import *
from kuka import Kuka
import numpy as np

#Connect to robot
robot = openshowvar(ip = '192.168.100.147', port = 7000)
kuka = Kuka(robot)

# Set base and tool frame
kuka.set_base(np.array([0, 0, 0, 0, 0, 0]))
kuka.set_tool(np.array([0, 0, 0, 0, 0, 0]))

# Read base and tool frame to confirm
kuka.read_base()
kuka.read_tool()
print("Base Frame: ", kuka.base_frame)
print("Tool Frame: ", kuka.tool_frame)

# Set tcp velocity (mm/s)
kuka.set_tool_velocity(10)

# Set the advance parameter to 5. Parameters tells the KUKA controller how many
# steps in advance it can process. It is needed for continuous movement
kuka.set_advance(5)

# Set the $APO.CDIS > 0 for continuous movement. It is the parameter telling
# the kuka controller when it is allowed to deviate from given path (mm).
kuka.robot.write("$APO.CDIS", str(100))
kuka.read_APO()

# Read current cartessian position and orientation of end-effector
kuka.read_cartessian()

# Construct trajectory
trajectory_arr = []


# Construct square trajectory
trajectory_arr.append(np.array([kuka.x_cartessian + 100, kuka.y_cartessian, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian]))
trajectory_arr.append(np.array([kuka.x_cartessian + 100, kuka.y_cartessian + 100, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian]))
trajectory_arr.append(np.array([kuka.x_cartessian, kuka.y_cartessian + 100, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian]))
trajectory_arr.append(np.array([kuka.x_cartessian, kuka.y_cartessian, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian]))
trajectory_arr.append(np.array([kuka.x_cartessian, kuka.y_cartessian, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian]))

trajectory_arr = np.array(trajectory_arr)

# Send trajectory
kuka.ptp_continuous(trajectory_arr)
