import numpy as np

class Kuka:
    def __init__(self, robot):
        self.robot = robot
        #Test connection
        if not self.robot.can_connect:
                print('Connection error')
                import sys
                sys.exit(-1)
        self.name = self.robot.read('$ROBNAME[]', debug=False).decode()

    def lin_continuous(self, arr, log=0):
        self.send_Frame_array(arr)
        self.robot.write("COM_LENGTH", str(arr.shape[0]-1), False)
        if log == 1:
            self.robot.write("COM_LOG", "1", False)
            self.robot.write("COM_CASEVAR", "5", False)
        else:
            self.robot.write("COM_CASEVAR", "4", False)

    def read_advance(self):
        self.advance = self.robot.read("$ADVANCE", False).decode()

    def read_APO(self):
        self.APO = self.robot.read("$APO", False).decode()

    def ptp(self, arr):
        self.send_Frame(arr, "COM_FRAME")
        self.robot.write("COM_CASEVAR", "1", False)

    def read_base(self):
        string = self.robot.read("$BASE", False).decode()
        string = string.replace(',', '')
        string = string.replace('{', '')
        string = string.replace('}', '')
        string = string.split()
        self.base_frame_x = float(string[2])
        self.base_frame_y = float(string[4])
        self.base_frame_z = float(string[6])
        self.base_frame_A = float(string[8])
        self.base_frame_B = float(string[10])
        self.base_frame_C = float(string[12])
        self.base_frame = np.array([self.base_frame_x, self.base_frame_y, self.base_frame_z, self.base_frame_A, self.base_frame_B, self.base_frame_C])

    def read_cartessian(self):
        cartessian_string = self.robot.read("$POS_ACT", False).decode()
        cartessian_string = cartessian_string.replace(',', '')
        cartessian = cartessian_string.split()
        self.x_cartessian = float(cartessian[2])
        self.y_cartessian = float(cartessian[4])
        self.z_cartessian = float(cartessian[6])
        self.A_cartessian = float(cartessian[8])
        self.B_cartessian = float(cartessian[10])
        self.C_cartessian = float(cartessian[12])
        self.E1_cartessian = float(cartessian[14])
        self.E2_cartessian = float(cartessian[16])
        self.E3_cartessian = float(cartessian[18])
        self.E4_cartessian = float(cartessian[20])
        self.E5_cartessian = float(cartessian[22])
        self.E6_cartessian = float(cartessian[24])
        self.global_position = np.array([self.x_cartessian, self.y_cartessian, self.z_cartessian, self.A_cartessian, self.B_cartessian, self.C_cartessian])


    def read_input(self, index):
        return(self.robot.read(("$IN[" + str(index) + "]"), False).decode())

    def read_out(self, index):
        return(self.robot.read(("$OUT[" + str(index) + "]"), False).decode())

    def read_tcp_velocity(self):
        self.velocity_cartessian = self.robot.read("$VEL_C", False).decode()

    def read_tool(self):
        string = self.robot.read("$TOOL", False).decode()
        string = string.replace(',', '')
        string = string.replace('{', '')
        string = string.replace('}', '')
        string = string.split()
        self.tool_frame_x = float(string[2])
        self.tool_frame_y = float(string[4])
        self.tool_frame_z = float(string[6])
        self.tool_frame_A = float(string[8])
        self.tool_frame_B = float(string[10])
        self.tool_frame_C = float(string[12])
        self.tool_frame = np.array([self.tool_frame_x, self.tool_frame_y, self.tool_frame_z, self.tool_frame_A, self.tool_frame_B, self.tool_frame_C])

    def send_E6POS(self, arr, system_variable=""):
        string_arr = []
        for i in range(len(arr)):
            string_arr.append(str(arr[i]))
        string = ("{E6POS: X " + string_arr[0] + ", Y " + string_arr[1] + ", Z "+ string_arr[2] + ", A " + string_arr[3] + ", B " + string_arr[4] + ", C " + string_arr[5] + ", E1 " + str(self.E1_cartessian) + ", E2 " + str(self.E2_cartessian) + ", E3 " + str(self.E3_cartessian) + ", E4 " + str(self.E4_cartessian) + ", E5 " + str(self.E5_cartessian) + ", E6 " + str(self.E6_cartessian) + "}")
        print("string to be sent: ", string, " variable: ", system_variable)
        self.robot.write(system_variable, string, False)

    def send_Frame(self, arr, system_variable=""):
        string_arr = []
        for i in range(len(arr)):
            string_arr.append(str(arr[i]))
        cartessian_string = ("{FRAME: X " + string_arr[0] + ", Y " + string_arr[1] + ", Z "+ string_arr[2] + ", A " + string_arr[3] + ", B " + string_arr[4] + ", C " + string_arr[5] + "}")
        print(cartessian_string)
        self.robot.write(system_variable, cartessian_string, False)

    def send_Frame_array(self, arr):
        #self.robot.write("COM_LENGTH", str(length)) # Send length of array
        for i in range(len(arr)):
            index_string = ("COM_FRAME_ARRAY[" + str(i) + "]")
            self.send_Frame(arr[i], index_string)

    def set_advance(self, value):
        self.robot.write("$ADVANCE", str(value), False)

    def set_APO_CPTP(self, value):
        self.robot.write("$APO.CPTP", str(value), False)

    def set_base(self, arr):
        self.send_Frame(arr, "$COM_FRAME")
        self.robot.write("COM_CASEVAR", "8")

    def set_input(self, index, bool):
        self.robot.write("COM_IDX", index, False)
        self.robot.write("COM_BOOL", bool, False)
        self.robot.write("COM_CASEVAR", "2", False)

    def set_output(self, index, bool):
        self.robot.write("COM_IDX", index, False)
        self.robot.write("COM_BOOL", bool, False)
        self.robot.write("COM_CASEVAR", "3", False)

    def set_tool(self, arr):self.send_Frame(arr, "$COM_FRAME")
        self.robot.write("COM_CASEVAR", "8")
        self.send_Frame(arr, "$COM_FRAME")
        self.robot.write("COM_CASEVAR", "7")

    def set_tool_velocity(self, value):
        self.robot.write("COM_VALUE1", str(value), False)
        self.robot.write("COM_CASEVAR", "6", False)
