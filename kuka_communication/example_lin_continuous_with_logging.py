# Work in progress

import sys
sys.path.insert(0, 'include/')

from openshowvar import *
from datetime import datetime
import time
import csv
from kuka import Kuka
import threading

def log_end_effector_cartessian(frequency = 83.3):

    while True:

        com_log = int(kuka.robot.read("COM_LOG").decode())

        if com_log == 1:

            filename=str(datetime.now())

            f= open(filename, "w+")
            time_to_process = 1/frequency

            kuka.robot.write("COM_LOG_READY", "1")

            while com_log == 1:

                com_log = int(kuka.robot.read("COM_LOG").decode())

                time_start = datetime.now()
                kuka.read_cartessian()
                kuka.read_tcp_velocity()
                out = kuka.read_out(1)
                f.write("%d," % i)
                f.write("%f," % kuka.x_cartessian)
                f.write("%f," % kuka.y_cartessian)
                f.write("%f," % kuka.z_cartessian)
                f.write("%f," % kuka.A_cartessian)
                f.write("%f," % kuka.B_cartessian)
                f.write("%f," % kuka.C_cartessian)
                f.write("%s," % out)
                #velocity - to do
                f.write("\n")

                #writer.writerow([i, datetime.now(), str(kuka.x_cartessian), str(kuka.y_cartessian), str(kuka.z_cartessian), str(kuka.A_cartessian), str(kuka.B_cartessian), str(kuka.C_cartessian), str(self.velocity_cartessian)]) # Write to csv file

                time_end = datetime.now()
                process_time = (time_end - time_start).microseconds*0.000001
                time.sleep(time_to_process-process_time) # Wait before logging next position

robot = openshowvar(ip = '192.168.100.147', port = 7000)

kuka = Kuka(robot)

#Test connection

if not robot.can_connect:
        print('Connection error')
        import sys
        sys.exit(-1)
#print('\nConnected KRC Name: ', end=' ')
robot_name = robot.read('$ROBNAME[]', False)


# Start logging thread
t1 = threading.Thread(target=log_end_effector_cartessian, args=(83,))
t1.start()


# Send trajectory here

# Set base and tool frame
kuka.set_base(np.array([0, 0, 0, 0, 0, 0]))
kuka.set_tool(np.array([0, 0, 0, 0, 0, 0]))

# Read base and tool frame to confirm
kuka.read_base()
kuka.read_tool()
print("Base Frame: ", kuka.base_frame)
print("Tool Frame: ", kuka.tool_frame)


# Set tcp velocity (mm/s)
kuka.set_tool_velocity(0.001)

# Set the advance parameter to 5. Parameters tells the KUKA controller how many
# steps in advance it can process. It is needed for continuous movement
kuka.set_advance(5)

# Set the $APO.CDIS > 0 for continuous movement. It is the parameter telling
# the kuka controller when it is allowed to deviate from given path (mm).
kuka.robot.write("$APO.CDIS", str(100))
kuka.read_APO()

# Read current cartessian position and orientation of end-effector
kuka.read_cartessian()


# Start logging thread
t1 = threading.Thread(target=log_end_effector_cartessian, args=(83,))
t1.start()

trajectory_arr = []

# Send trajectory here
trajectory_arr.append(np.array([kuka.x_cartessian, kuka.y_cartessian + 10, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian]))
trajectory_arr.append(np.array([kuka.x_cartessian, kuka.y_cartessian + 70, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian]))
trajectory_arr.append(np.array([kuka.x_cartessian, kuka.y_cartessian, kuka.z_cartessian, kuka.A_cartessian, kuka.B_cartessian, kuka.C_cartessian]))

trajectory_arr = np.array(trajectory_arr)

# Send trajectory
kuka.lin_continuous(trajectory_arr, log=1)

# Wait for logging thread to exit, but it never will.
t1.join()

#WAIT FOR $VEL_ACT==0
try:
    pass
except Exception as e:
    raise
