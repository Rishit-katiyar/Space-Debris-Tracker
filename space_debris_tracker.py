
import cv2
import csv
import numpy as np
from datetime import datetime
from threading import Thread
import matplotlib.pyplot as plt

class SpaceDebrisTracker:
    def __init__(self, telescope_index=0):
        self.telescope_index = telescope_index
        self.cap = cv2.VideoCapture(self.telescope_index)
        self.frame_number = 0
        self.debris_data = []
        self.debris_counter = 0
        self.thread_active = False
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.min_area = 50
        self.max_area = 1000
        self.visualization_enabled = True

    def count_and_track_debris(self, frame):
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

    def update_debris_info(self):
        while self.thread_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_number += 1

            debris_info = self.count_and_track_debris(frame)
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
        self.thread_active = True
        t = Thread(target=self.update_debris_info)
        t.start()

    def stop_tracking(self):
        self.thread_active = False
        self.cap.release()
        cv2.destroyAllWindows()

    def save_debris_data(self, filename='debris_positions.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['FrameNumber', 'DebrisName', 'DetectionTime', 'X', 'Y'])
            for debris_info in self.debris_data:
                writer.writerow(debris_info)

    def plot_debris_counts(self):
        frame_numbers = [debris[0] for debris in self.debris_data]
        unique_frame_numbers, counts = np.unique(frame_numbers, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique_frame_numbers, counts, color='skyblue')
        plt.xlabel('Frame Number')
        plt.ylabel('Debris Count')
        plt.title('Debris Count per Frame')
        plt.grid(True)
        plt.show()

    def display_menu(self):
        print("Space Debris Tracker Menu:")
        print("1. Toggle Visualization")
        print("2. Save Debris Data")
        print("3. Plot Debris Counts")
        print("4. Set Min Area")
        print("5. Set Max Area")
        print("6. Exit")

    def start_menu(self):
        self.start_tracking()
        while True:
            self.display_menu()
            choice = input("Enter your choice (1-6): ")
            if choice == '1':
                self.toggle_visualization()
            elif choice == '2':
                self.save_debris_data()
            elif choice == '3':
                self.plot_debris_counts()
            elif choice == '4':
                min_area = int(input("Enter minimum area: "))
                self.set_min_area(min_area)
            elif choice == '5':
                max_area = int(input("Enter maximum area: "))
                self.set_max_area(max_area)
            elif choice == '6':
                self.stop_tracking()
                break
            else:
                print("Invalid choice. Please select a number from 1 to 6.")

# Example usage:
if __name__ == "__main__":
    debris_tracker = SpaceDebrisTracker(telescope_index=0)
    debris_tracker.start_menu()
