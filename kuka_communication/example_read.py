import sys
sys.path.insert(0, 'include/')

from openshowvar import *
from kuka import Kuka
import numpy as np

robot = openshowvar(ip = '192.168.100.147', port = 7000)
kuka = Kuka(robot)
print(robot.can_connect)

print(robot.read("$OV_PRO"))
print()
print(robot.read("$POS_ACT"))
print()
print(robot.read("$OUT[1]").decode())

print()
print(robot.read("$IN[1]").decode())

print()
print(robot.read("COM_CASEVAR", False).decode())


print()
print(robot.read("COM_IDX", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[2]", False).decode())


print()
print(kuka.robot.read("COM_FRAME_ARRAY[1]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[2]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[3]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[4]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[5]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[6]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[7]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[8]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[9]", False).decode())

print()
print(kuka.robot.read("COM_FRAME_ARRAY[10]", False).decode())

print()
print(kuka.robot.read("COM_LENGTH", False).decode())

print()
print(kuka.robot.read("COM_CASEVAR", False).decode())

print()
print(kuka.robot.read("Counter_variable", False).decode())

print()
kuka.read_cartessian()
print(kuka.global_position)

kuka.read_advance()
print(kuka.advance)

kuka.read_APO()
print(kuka.APO)
