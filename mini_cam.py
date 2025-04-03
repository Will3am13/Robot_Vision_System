import cv2
import numpy as np
import math
import subprocess
import time
import threading

def find_centroid(mask):
    M = cv2.moments(mask)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] else None

def calculate_distance(centroid, frame_center):
    if not centroid:
        return None, None, None
    x_diff, y_diff = centroid[0] - frame_center[0], centroid[1] - frame_center[1]
    return x_diff, y_diff, math.sqrt(x_diff**2 + y_diff**2)

def toggle_usb_power(turn_on):
    try:
        action = "on" if turn_on else "off"
        subprocess.run(['uhubctl', '-a', action], check=True)
        print(f"USB power {action.upper()}")
    except Exception as e:
        print(f"Error toggling USB power: {e}")

def power_cycle():
    print("Power cycling USB...")
    toggle_usb_power(False)
    time.sleep(3)
    toggle_usb_power(True)
    time.sleep(5)

def initialize_camera():
    for _ in range(3):
        cap = cv2.VideoCapture(0)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
        power_cycle()
    return cv2.VideoCapture(0)

def main():
    while True:
        cap = initialize_camera()
        threshold = 120
        print("Press '+'/'-' to adjust threshold, 'p' to reset USB, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error, power cycling...")
                cap.release()
                power_cycle()
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            centroid = find_centroid(mask)
            
            if centroid:
                x_diff, y_diff, distance = calculate_distance(centroid, (frame.shape[1]//2, frame.shape[0]//2))
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Dist: {distance:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('+'):
                threshold = min(threshold + 5, 255)
            elif key == ord('-'):
                threshold = max(threshold - 5, 0)
            elif key == ord('p'):
                cap.release()
                power_cycle()
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
