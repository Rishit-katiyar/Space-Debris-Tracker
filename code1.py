import cv2
import csv
import numpy as np
from datetime import datetime
from threading import Thread, Lock
import matplotlib.pyplot as plt
import configparser
import logging
import os
from cryptography.fernet import Fernet
import json

# Configuration for logging
logging.basicConfig(filename='debris_tracker.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Encryption utilities
def load_key():
    """
    Load the encryption key from a file.
    If the file does not exist, create a new key and save it.
    """
    if not os.path.exists("secret.key"):
        key = Fernet.generate_key()
        with open("secret.key", "wb") as key_file:
            key_file.write(key)
    return open("secret.key", "rb").read()

def encrypt_message(message):
    """
    Encrypt a message using the loaded encryption key.
    """
    key = load_key()
    f = Fernet(key)
    return f.encrypt(message.encode())

def decrypt_message(encrypted_message):
    """
    Decrypt a message using the loaded encryption key.
    """
    key = load_key()
    f = Fernet(key)
    return f.decrypt(encrypted_message).decode()

# Configuration management
config = configparser.ConfigParser()
config.read('debris_tracker.ini')

class SpaceDebrisTracker:
    """
    Space Debris Tracker class for tracking and analyzing space debris using video feed.
    """

    def __init__(self, telescope_index=0):
        """
        Initialize the SpaceDebrisTracker.
        """
        self.telescope_index = telescope_index
        self.cap = cv2.VideoCapture(self.telescope_index)
        self.frame_number = 0
        self.debris_data = []
        self.debris_counter = 0
        self.thread_active = False
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.min_area = int(config['Detection']['MinArea'])
        self.max_area = int(config['Detection']['MaxArea'])
        self.visualization_enabled = config['Visualization'].getboolean('Enabled')
        self.lock = Lock()

    def count_and_track_debris(self, frame):
        """
        Count and track space debris in the given frame.
        """
        try:
            fg_mask = self.background_subtractor.apply(frame)
            fg_mask = cv2.medianBlur(fg_mask, 5)
            fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            debris_positions = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        debris_positions.append((
                            self.frame_number,
                            "Space Debris",
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            cX,
                            cY
                        ))
                        self.debris_counter += 1

            return debris_positions
        except Exception as e:
            logging.error(f"Error in count_and_track_debris: {e}")
            return []

    def update_debris_info(self):
        """
        Update the debris information by continuously processing the video feed.
        """
        while self.thread_active:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Frame capture failed")
                break

            self.frame_number += 1

            debris_info = self.count_and_track_debris(frame)
            with self.lock:
                self.debris_data.extend(debris_info)

            if self.visualization_enabled:
                for debris in debris_info:
                    frame_num, debris_name, detection_time, x, y = debris
                    cv2.putText(frame, f"{debris_name}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

                cv2.imshow('Space Debris Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def start_tracking(self):
        """
        Start the debris tracking process in a new thread.
        """
        self.thread_active = True
        t = Thread(target=self.update_debris_info)
        t.start()

    def stop_tracking(self):
        """
        Stop the debris tracking process.
        """
        self.thread_active = False
        self.cap.release()
        cv2.destroyAllWindows()

    def save_debris_data(self, filename='debris_positions.csv'):
        """
        Save the tracked debris data to a CSV file with encryption.
        """
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['FrameNumber', 'DebrisName', 'DetectionTime', 'X', 'Y'])
                for debris_info in self.debris_data:
                    encrypted_data = [encrypt_message(str(item)) for item in debris_info]
                    writer.writerow(encrypted_data)
            logging.info("Debris data saved successfully")
        except Exception as e:
            logging.error(f"Error in save_debris_data: {e}")

    def load_debris_data(self, filename='debris_positions.csv'):
        """
        Load the tracked debris data from a CSV file with decryption.
        """
        try:
            with open(filename, mode='r', newline='') as file:
                reader = csv.reader(file)
                self.debris_data = []
                next(reader)  # Skip header
                for row in reader:
                    decrypted_data = [decrypt_message(item) for item in row]
                    self.debris_data.append(tuple(decrypted_data))
            logging.info("Debris data loaded successfully")
        except Exception as e:
            logging.error(f"Error in load_debris_data: {e}")

    def plot_debris_counts(self):
        """
        Plot the debris counts per frame.
        """
        try:
            frame_numbers = [int(debris[0]) for debris in self.debris_data]
            unique_frame_numbers, counts = np.unique(frame_numbers, return_counts=True)
            plt.figure(figsize=(10, 6))
            plt.bar(unique_frame_numbers, counts, color='skyblue')
            plt.xlabel('Frame Number')
            plt.ylabel('Debris Count')
            plt.title('Debris Count per Frame')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error in plot_debris_counts: {e}")

    def toggle_visualization(self):
        """
        Toggle the visualization of the tracking process.
        """
        self.visualization_enabled = not self.visualization_enabled
        logging.info(f"Visualization {'enabled' if self.visualization_enabled else 'disabled'}.")

    def set_min_area(self, min_area):
        """
        Set the minimum area for debris detection.
        """
        self.min_area = min_area
        logging.info(f"Minimum area set to {self.min_area}.")

    def set_max_area(self, max_area):
        """
        Set the maximum area for debris detection.
        """
        self.max_area = max_area
        logging.info(f"Maximum area set to {self.max_area}.")

    def display_menu(self):
        """
        Display the user menu for interacting with the debris tracker.
        """
        print("Space Debris Tracker Menu:")
        print("1. Toggle Visualization")
        print("2. Save Debris Data")
        print("3. Load Debris Data")
        print("4. Plot Debris Counts")
        print("5. Set Min Area")
        print("6. Set Max Area")
        print("7. Exit")

    def start_menu(self):
        """
        Start the user menu for interacting with the debris tracker.
        """
        self.start_tracking()
        while True:
            self.display_menu()
            choice = input("Enter your choice (1-7): ")
            if choice == '1':
                self.toggle_visualization()
            elif choice == '2':
                self.save_debris_data()
            elif choice == '3':
                self.load_debris_data()
            elif choice == '4':
                self.plot_debris_counts()
            elif choice == '5':
                min_area = int(input("Enter minimum area: "))
                self.set_min_area(min_area)
            elif choice == '6':
                max_area = int(input("Enter maximum area: "))
                self.set_max_area(max_area)
            elif choice == '7':
                self.stop_tracking()
                break
            else:
                print("Invalid choice. Please select a number from 1 to 7.")
                logging.warning("Invalid menu choice")

# Ensure the encryption key is available
if not os.path.exists("secret.key"):
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)

# Example usage:
if __name__ == "__main__":
    debris_tracker = SpaceDebrisTracker(telescope_index=0)
    debris_tracker.start_menu()
