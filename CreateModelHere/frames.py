import cv2
import os

def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    existing_frames = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(output_folder) if f.startswith("frame_") and f.endswith(".jpg")]
    frame_count = max(existing_frames) + 1 if existing_frames else 0
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        frame_count += 1
    
    video_capture.release()
    print("Done!")

video_path = 'video/normal.mp4'
video_path2 = 'video/normal2.mp4'
video_path3 = 'video/normal3.mp4'
output_folder = 'dataset/correct'

video_to_frames(video_path, output_folder)
video_to_frames(video_path2, output_folder)
video_to_frames(video_path3, output_folder)
