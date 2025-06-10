import pygame
import pyvesc

def run_motor(speed, steering, motor):
    motor.set_duty_cycle(speed)

    servo_pos = (steering + 1) / 2
    motor.set_servo(servo_pos)

def initialize_controller(joystick):
    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    print(f"joystick count: {joystick_count}")
    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        return joystick
    return None

def capture_brake(joystick):
    brake = joystick.get_axis(2)
    brake = (brake + 1) / 2
    return brake * -1

def capture_acceleration(joystick):
    acceleration = joystick.get_axis(5)
    acceleration = (acceleration + 1) / 2
    return acceleration

def capture_steering(joystick):
    steering = joystick.get_axis(0)
    steering = max(-1.0, min(1, steering))
    if -0.1 <= steering <= 0.1:
        steering = 0
    return steering

def capture_controller(joystick, serial_port, speed_limit):
    motor = pyvesc.VESC(serial_port)
    steering = None
    speed = None
    brake = None
    try:
        while True:
            pygame.event.pump()
            if joystick:
                steering = capture_steering(joystick)
                brake = capture_brake(joystick)
                speed = capture_acceleration(joystick) + capture_brake(joystick)

                if speed > speed_limit / 100 :
                    speed = speed_limit / 100
                if speed < -0.15 :
                    speed = 0.15
                print(f"steering: {steering}, brake: {brake}, speed: {speed}")
                run_motor(speed, steering, motor)
    finally:
        motor.close()

def main():
    serial_port = '/dev/ttyACM0'
    speed_limit = 30
    joystick = None
    joystick = initialize_controller(joystick)
    capture_controller(joystick, serial_port, speed_limit)

if __name__ == '__main__':
    exit(main())

 
