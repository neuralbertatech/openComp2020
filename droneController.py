import time
import tellopy
import sys
import threading
import queue
import os

SPEED = 20 # value 0-100

# To run this in two terminals, type "tty" in one terminal and in the other, type "Tellotest.py > [result of tty command]"

def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)

def printWelcomeMessage():
    print("\nThe drone has been connected!\n")
    print("Controls:")
    print("Up - u")
    print("Down - d")
    print("Forward - f")
    print("Backwards - b")
    print("Left - l")
    print("Right - r")
    print("Rotate Left - rl")
    print("Rotate Right - rr\n")

    print("Special Controls:")
    print("Takeoff - tkoff")
    print("Flip - flip")
    print("Land - lnd")
    print("Kill - k\n\n")



def connectDrone():
    drone = tellopy.Tello()
    try:
        # Connect Drone
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()
        drone.wait_for_connection(60.0)
        printWelcomeMessage()

        # Drone Connected! Begin BCI control
        # Note that the BCI controls the interval speed, this program simply
        # controls the drone
        while(True):
            command = input("Command: ") #TODO replace these with the BCI controller
            updateDrone(drone, command, SPEED)

    except Exception as ex:
        print(ex)
    finally:
        drone.quit()

def updateDrone(drone, command, speed):
        # Resets the drone's movement
        drone.up(0)
        drone.down(0)
        drone.forward(0)
        drone.backward(0)
        drone.left(0)
        drone.right(0)
        drone.clockwise(0)
        drone.counter_clockwise(0)

        ##TODO Add an if bypass to check if the same argument is passed in twice, reduces jitter


        if(command == "tkoff"):
            drone.takeoff()
        elif(command  == "u"):
            drone.up(speed)
        elif(command == "d"):
            drone.down(speed)
        elif(command == "f"):
            drone.forward(speed)
        elif(command == "b"):
            drone.backward(speed)
        elif(command == "l"):
            drone.left(speed)
        elif(command == "r"):
            drone.right(speed)
        elif(command == "rl"):
            drone.clockwise(speed)
        elif(command == "rr"):
            drone.counter_clockwise(speed)
        elif(command == "ffr"):
            drone.flip_forwardright()
        elif(command == "lnd"):
            drone.land()
        elif(command == "k"):
            if(input("Are you sure you want to kill the drone? (y/n)") == "y"):
                drone.emergency()

if __name__ == '__main__':
    connectDrone()
