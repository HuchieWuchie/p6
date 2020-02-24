import sys
sys.path.insert(0, 'include/')

from py_openshowvar import *
from kuka import Kuka
import numpy as np
import getch
import time


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

# Read tcp
kuka.read_cartessian()

while True:

    # Get command
    key = getch.getch()

    if key != None:
        # Read current cartessian position and orientation of end-effector
        kuka.read_cartessian()
    desired_point = np.array([kuka.x_cartessian, kuka.y_cartessian, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian])

    if key == 'q': # x positive
        x = kuka.x_cartessian + 10
        desired_point = np.array([x, kuka.y_cartessian, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian])
    elif key == 'w': # x negative
        x = kuka.x_cartessian - 10
        desired_point = np.array([x, kuka.y_cartessian, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian])
    elif key == 'a':
        y = kuka.y_cartessian + 10
        desired_point = np.array([kuka.x_cartessian, y, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian])
    elif key == 's':
        y = kuka.y_cartessian - 10
        desired_point = np.array([kuka.x_cartessian, y, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian])
    elif key == 'z':
        z = kuka.z_cartessian + 10
        desired_point = np.array([kuka.x_cartessian, kuka.y_cartessian, z, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian])
    elif key == 'x':
        z = kuka.z_cartessian - 10
        desired_point = np.array([kuka.x_cartessian, kuka.y_cartessian, z, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian])



    if key != None:
        # Set end effector destination position

        print("Sending")
        # Send ptp command
        kuka.ptp(desired_point)

    #time.sleep(0.05)
