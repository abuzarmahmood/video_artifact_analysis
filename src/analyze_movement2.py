import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import argparse
from pathlib import Path
from remove_background import remove_background
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

def reduce_dimensionality_incremental(video_path, n_components=1, batch_size=100):
    """
    Perform incremental PCA dimensionality reduction on frame differences.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # Process frames in batches
    batch = []
    embedding = []
    prev_gray = None
    
    pbar = tqdm(total=total_frames-1, desc="Processing frames")  # -1 because we're using differences
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to grayscale and normalize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        if prev_gray is not None:
            # Compute frame difference
            diff = gray - prev_gray
            batch.append(diff.flatten())
            pbar.update(1)
            
            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                batch_array = np.array(batch)
                ipca.partial_fit(batch_array)
                embedding.extend(ipca.transform(batch_array))
                batch = []
        
        prev_gray = gray
    
    # Process remaining frames
    if batch:
        batch_array = np.array(batch)
        ipca.partial_fit(batch_array)
        embedding.extend(ipca.transform(batch_array))
    
    cap.release()
    return np.array(embedding), total_frames - 1  # -1 because we're using differences

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
    plt.plot(time_seconds, np.abs(embedding[:, 0]), '-b', alpha=0.7)
    plt.yscale('log')
    plt.title('Movement Trajectory (Incremental PCA 1D Projection)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('PCA Component 1 (log scale)')
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze movement in video using frame differences and PCA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input', type=str,
                       help='Path to input video file')
    
    # Output options
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument('--output-dir', type=str, default='output',
                          help='Directory for saving results')
    
    # Analysis parameters
    analysis_group = parser.add_argument_group('Analysis parameters')
    analysis_group.add_argument('--batch-size', type=int, default=100,
                            help='Batch size for incremental PCA processing')
    analysis_group.add_argument('--n-components', type=int, default=1,
                            help='Number of PCA components to compute')
    analysis_group.add_argument('--min-frames', type=int, default=30,
                            help='Minimum number of frames with movement to include in mask')
    analysis_group.add_argument('--max-frames', type=int, default=1000,
                            help='Maximum number of frames to process (0 for all frames)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get video properties
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print("Performing incremental dimensionality reduction...")
    # Create occupancy mask first
    mask, _ = create_occupancy_mask(
        args.input,
        min_frames=args.min_frames,
        max_frames=args.max_frames
    )
    
    embedding, total_frames = reduce_dimensionality_incremental(
        args.input,
        n_components=args.n_components,
        batch_size=args.batch_size
    )
    
    # Save results
    plot_path = output_dir / 'movement_trajectory.png'
    print(f"Saving trajectory plot to {plot_path}")
    plot_results(embedding, fps, plot_path)
    
    print(f'Processed {total_frames} frames ({total_frames/fps:.1f} seconds)')
    
    # Save and plot mask
    mask_path = output_dir / 'occupancy_mask.npy'
    mask_plot_path = output_dir / 'occupancy_mask.png'
    print(f"Saving occupancy mask plot to {mask_plot_path}")
    plot_occupancy_mask(mask, mask_plot_path)
    np.save(mask_path, mask)
    
    # Save embedding
    embedding_path = output_dir / 'embedding.npy'
    np.save(embedding_path, embedding)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
