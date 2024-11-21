import cv2
import mediapipe as mp
import numpy as np
import argparse

def remove_background(input_video, output_video, threshold=0.5):
    # Initialize MediaPipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = selfie_segmentation.process(rgb_frame)
        
        # Generate mask
        mask = results.segmentation_mask > threshold
        mask = np.stack((mask,) * 3, axis=-1)
        
        # Create transparent background
        output_frame = np.zeros_like(frame)
        output_frame[mask] = frame[mask]

        # Write the frame
        out.write(output_frame)
        
        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")

    # Release everything
    cap.release()
    out.release()
    selfie_segmentation.close()

def main():
    parser = argparse.ArgumentParser(description='Remove background from video')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', help='Output video file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Segmentation threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    print(f"Processing video: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    remove_background(args.input, args.output, args.threshold)
    print("Processing complete!")

if __name__ == "__main__":
    main()
