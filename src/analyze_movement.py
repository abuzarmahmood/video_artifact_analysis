import cv2
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import argparse
from pathlib import Path
from remove_background import remove_background

def create_occupancy_mask(video_path, threshold=0.1, min_frames=30):
    """
    Create an occupancy mask from a background-subtracted video.
    Returns both the mask and the weighted pixel data for dimensionality reduction.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize accumulator for occupancy
    occupancy = np.zeros((height, width), dtype=np.float32)
    frame_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to grayscale and normalize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Update occupancy map
        movement = gray > threshold
        occupancy += movement.astype(np.float32)
        
        # Store frame data for later analysis
        frame_data.append(gray.flatten())
    
    cap.release()
    
    # Normalize occupancy to [0,1]
    occupancy = occupancy / total_frames
    
    # Create binary mask where movement occurred in at least min_frames
    mask = occupancy > (min_frames / total_frames)
    
    return mask, np.array(frame_data)

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

def plot_results(embedding, output_path):
    """
    Plot the 1D embedding results over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(embedding[:, 0], '-b', alpha=0.7)
    plt.title('Movement Trajectory (UMAP 1D Projection)')
    plt.xlabel('Frame Number')
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
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process input video
    print("Creating occupancy mask...")
    mask, frame_data = create_occupancy_mask(args.input, 
                                           threshold=args.threshold,
                                           min_frames=args.min_frames)
    
    print("Performing dimensionality reduction...")
    embedding = reduce_dimensionality(frame_data, mask)
    
    # Save results
    plot_path = output_dir / 'movement_trajectory.png'
    print(f"Saving plot to {plot_path}")
    plot_results(embedding, plot_path)
    
    # Save mask
    mask_path = output_dir / 'occupancy_mask.npy'
    np.save(mask_path, mask)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
