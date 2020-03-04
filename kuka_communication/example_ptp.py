import sys
sys.path.insert(0, 'include/')

from openshowvar import *
from kuka import Kuka
import numpy as np

#Connect robot
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

# Read current cartessian position and orientation of end-effector
kuka.read_cartessian()

# Set end effector destination position
x_desired = kuka.x_cartessian - 150
y_desired = kuka.y_cartessian - 150
z_desired = kuka.z_cartessian - 20
desired_point = np.array([x_desired, y_desired, z_desired, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian])

# Send ptp command
kuka.ptp(desired_point)
