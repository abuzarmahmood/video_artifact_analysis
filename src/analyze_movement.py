import cv2
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import argparse
from pathlib import Path
from remove_background import remove_background

def create_occupancy_mask(video_path, threshold=0.1, min_frames=30, max_frames=1000):
    """
    Create an occupancy mask from a background-subtracted video.
    Returns both the mask and the weighted pixel data for dimensionality reduction.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize accumulator for occupancy
    occupancy = np.zeros((height, width), dtype=np.float32)
    frame_data = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames if we're over max_frames
        if max_frames and frame_count >= max_frames:
            break
            
        # Convert frame to grayscale and normalize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Update occupancy map
        movement = gray > threshold
        occupancy += movement.astype(np.float32)
        
        # Store frame data for later analysis
        frame_data.append(gray.flatten())
        frame_count += 1
    
    cap.release()
    
    # Normalize occupancy to [0,1]
    occupancy = occupancy / total_frames
    
    # Create binary mask where movement occurred in at least min_frames
    mask = occupancy > (min_frames / total_frames)
    
    return mask, np.array(frame_data), fps

def reduce_dimensionality(data, mask, n_components=1):
    """
    Perform UMAP dimensionality reduction on the masked pixel data.
    """
    # Flatten mask and use it to select relevant pixels
    flat_mask = mask.flatten()
    masked_data = data[:, flat_mask]
    
    # Initialize and fit UMAP
    reducer = UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(masked_data)
    
    return embedding

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

def plot_results(embedding, fps, output_path):
    """
    Plot the 1D embedding results over time.
    """
    time_seconds = np.arange(len(embedding)) / fps
    plt.figure(figsize=(12, 6))
    plt.plot(time_seconds, embedding[:, 0], '-b', alpha=0.7)
    plt.title('Movement Trajectory (UMAP 1D Projection)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('UMAP Dimension 1')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze movement in background-subtracted video')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('--threshold', type=float, default=0.1,
                      help='Threshold for movement detection (0.0-1.0)')
    parser.add_argument('--min-frames', type=int, default=30,
                      help='Minimum frames for occupancy mask')
    parser.add_argument('--max-frames', type=int, default=1000,
                      help='Maximum frames to use for analysis (0 for no limit)')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process input video
    print("Creating occupancy mask...")
    mask, frame_data, fps = create_occupancy_mask(args.input, 
                                                 threshold=args.threshold,
                                                 min_frames=args.min_frames,
                                                 max_frames=args.max_frames)
    
    print("Performing dimensionality reduction...")
    embedding = reduce_dimensionality(frame_data, mask)
    
    # Save results
    plot_path = output_dir / 'movement_trajectory.png'
    print(f"Saving trajectory plot to {plot_path}")
    plot_results(embedding, fps, plot_path)
    
    # Save and plot mask
    mask_path = output_dir / 'occupancy_mask.npy'
    mask_plot_path = output_dir / 'occupancy_mask.png'
    print(f"Saving occupancy mask plot to {mask_plot_path}")
    plot_occupancy_mask(mask, mask_plot_path)
    np.save(mask_path, mask)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
