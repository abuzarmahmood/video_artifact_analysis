import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

def create_occupancy_mask(video_path, threshold=0.1, min_frames=30, max_frames=1000):
    """
    Create an occupancy mask from a background-subtracted video using running statistics.

    Inputs:
        - video_path: path to input video file
        - threshold: threshold for detecting movement (default 0.1)
        - min_frames: minimum number of frames with movement to include in mask (default 30)
        - max_frames: maximum number of frames to process (default 1000, set to 0 to process all frames)

    Outputs:
        - mask: binary mask where movement occurred in at least min_frames
        - fps: frames per second of the input video
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize accumulator for occupancy
    occupancy = np.zeros((height, width), dtype=np.float32)
    frame_count = 0
    
    while cap.isOpened() and (frame_count <= max_frames or max_frames == 0):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to grayscale and normalize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Update occupancy map using running mean
        movement = gray > threshold
        occupancy = (frame_count * occupancy + movement.astype(np.float32)) / (frame_count + 1)
        frame_count += 1
    
    cap.release()
    
    # Create binary mask where movement occurred in at least min_frames
    mask = occupancy > (min_frames / frame_count)
    
    return mask, fps

def plot_occupancy_mask(mask, output_path):
    """
    Plot the occupancy mask as a heatmap.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(mask, cmap='hot')
    plt.colorbar(label='Occupancy')
    plt.title('Movement Occupancy Mask')
    plt.savefig(output_path)
    plt.close()

def apply_mask_to_video(input_path, output_path, mask):
    """Apply occupancy mask to video and save result"""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply mask to each color channel
        masked_frame = frame.copy()
        for c in range(3):
            masked_frame[:,:,c] = frame[:,:,c] * mask
            
        out.write(masked_frame)
    
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(
        description='Create and apply occupancy mask to video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input', type=str,
                       help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory for saving results')
    parser.add_argument('--min-frames', type=int, default=30,
                       help='Minimum number of frames with movement to include in mask')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum number of frames to process (0 for all frames)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Creating occupancy mask...")
    mask, fps = create_occupancy_mask(
        args.input,
        min_frames=args.min_frames,
        max_frames=args.max_frames
    )
    
    # Save mask and plot
    mask_path = output_dir / 'occupancy_mask.npy'
    mask_plot_path = output_dir / 'occupancy_mask.png'
    masked_video_path = output_dir / 'masked_video.mp4'
    
    print(f"Saving occupancy mask plot to {mask_plot_path}")
    plot_occupancy_mask(mask, mask_plot_path)
    np.save(mask_path, mask)
    
    print(f"Creating masked video at {masked_video_path}")
    apply_mask_to_video(args.input, str(masked_video_path), mask)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
