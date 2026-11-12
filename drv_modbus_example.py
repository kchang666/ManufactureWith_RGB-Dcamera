from drv_modbus import send
from drv_modbus import request
from pymodbus.client import ModbusTcpClient
import time

c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
c.connect()

home = [408.285, 0.0, 680.0120000000001, 178.969, -0.241, -103.145]

#x : 388.442, y : -73.33200000000001, z : 325.693, rx: 179.987, ry: -0.004, rz: -106.121
#x : 637.4350000000001, y : 221.607, z : 322.593, rx: 179.987, ry: -0.004, rz: -106.121
#z : 324.796

#send.Suction_ON(c)
send.gripper_ON(c)
time.sleep(2)

#send.Go_Position(c, 353.208, -48.09, 680.0, 179.987, -0.004, -106.121, 50) #home
#send.Go_Position(c, 388.442+100, -73.332+100, 680.0-200, 179.987, -0.004, -106.121, 50)
#中點
send.Go_Position(c, 398.367, -3.011, 500, 178.869, 1.053, 132.036, 50)
#send.Suction_ON(c)
send.gripper_ON(c)
time.sleep(2)
send.gripper_OFF(c)
#send.Suction_OFF(c)
time.sleep(2)

#POS 1
send.Go_Position(c, 263.18,-145.766,500.00,178.869,1.053,132.036, 70)
send.Go_Position(c, 263.18,-145.766,409.66700000000003,178.869,1.053,132.036, 70)
#263.18,-145.766,409.66700000000003,178.869,1.053,132.036
send.gripper_ON(c)
time.sleep(2)
send.gripper_OFF(c)
time.sleep(2)

#POS 2
send.Go_Position(c, 503.075,144.214,500.00,178.869,1.053,132.036, 70)
send.Go_Position(c, 503.075,144.214,409.66700000000003,178.869,1.053,132.036, 70)
#x : 503.075, y : 144.214, z : 409.66700000000003, rx: 178.869, ry: 1.053, rz: 132.036
send.gripper_ON(c)
time.sleep(2)
send.gripper_OFF(c)
time.sleep(2)
#POS 3
send.Go_Position(c,  484.31, -149.68, 480.862, 178.869, 1.053, 132.036,50)
send.Go_Position(c,  484.31, -149.68, 446.862, 178.869, 1.053, 132.036,50)
send.gripper_ON(c)
time.sleep(2)
send.Go_Position(c,  484.31, -149.68, 500.862, 178.869, 1.053, 132.036,50)
time.sleep(2)
send.Go_Position(c, 398.367, -3.011, 500, 178.869, 1.053, 132.036, 50)


while True:
    x , y , z , rx, ry, rz = request.Get_TCP_Pose(c)

    print("x : " + str(x ), end=", ")
    print("y : " + str(y ), end=", ")
    print("z : " + str(z ), end=", ")
    print("rx: " + str(rx), end=", ")
    print("ry: " + str(ry), end=", ")
    print("rz: " + str(rz))

    time.sleep(0.2)

c.close()