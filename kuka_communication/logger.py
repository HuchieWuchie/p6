# Work in progress

import sys
sys.path.insert(0, 'include/')

from py_openshowvar import *
from datetime import datetime
import time
import csv
from kuka import Kuka

def log_end_effector_cartessian(filename, frequency = 83.3):

    f= open(filename, "w+")
    time_to_process = 1/frequency

    for i in range(100):
        time_start = datetime.now()
        kuka.read_cartessian()
        kuka.read_tcp_velocity()
        f.write("%d," % i)
        f.write("%f," % kuka.x_cartessian)
        f.write("%f," % kuka.y_cartessian)
        f.write("%f," % kuka.z_cartessian)
        f.write("%f," % kuka.A_cartessian)
        f.write("%f," % kuka.B_cartessian)
        f.write("%f," % kuka.C_cartessian)
        f.write("\n")

        #writer.writerow([i, datetime.now(), str(kuka.x_cartessian), str(kuka.y_cartessian), str(kuka.z_cartessian), str(kuka.A_cartessian), str(kuka.B_cartessian), str(kuka.C_cartessian), str(self.velocity_cartessian)]) # Write to csv file

        time_end = datetime.now()
        process_time = (time_end - time_start).microseconds*0.000001
        time.sleep(time_to_process-process_time) # Wait before logging next position



robot = openshowvar(ip = '192.168.100.147', port = 7000)

kuka = Kuka(robot)

#Test connection
"""
if not robot.can_connect:
        print('Connection error')
        import sys
        sys.exit(-1)
#print('\nConnected KRC Name: ', end=' ')
robot_name = robot.read('$ROBNAME[]', False)
"""

log_end_effector_cartessian(filename=str(datetime.now()))
