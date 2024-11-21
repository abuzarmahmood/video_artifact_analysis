import cv2
import argparse
from pathlib import Path

def get_video_resolution(video_path):
    """Get the resolution of a video file"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def downsample_video(input_path, output_path, downsample_factor):
    """Downsample video by the given factor"""
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions
    new_width = int(width / downsample_factor)
    new_height = int(height / downsample_factor)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        out.write(resized_frame)
    
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description='Downsample video resolution')
    parser.add_argument('input_path', type=str, help='Path to input video')
    parser.add_argument('output_path', type=str, help='Path to save downsampled video')
    args = parser.parse_args()
    
    # Get and display current resolution
    width, height = get_video_resolution(args.input_path)
    print(f"Current video resolution: {width}x{height}")
    
    # Get downsample factor from user
    while True:
        try:
            factor = float(input("Enter downsample factor (e.g., 2.0 will halve the resolution): "))
            if factor <= 0:
                print("Factor must be positive")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    print(f"New resolution will be: {int(width/factor)}x{int(height/factor)}")
    confirm = input("Continue? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Operation cancelled")
        return
        
    downsample_video(args.input_path, args.output_path, factor)
    print("Video downsampling complete!")

if __name__ == "__main__":
    main()
