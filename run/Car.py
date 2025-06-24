##
## EPITECH PROJECT, 2025
## Car
## File description:
## Car
##

import pyvesc

class Car:
    def __init__(self, serial_port):
        self.motor = pyvesc.VESC(serial_port)
        steering = None
        speed = None
    
    def setSpeed(self, speed):
        self.motor.set_duty_cycle(speed)

    def setSteering(self, steeringAngle):
        self.motor.set_servo(steeringAngle)

    def destroy(self):
        self.motor.stop_heartbeat()

