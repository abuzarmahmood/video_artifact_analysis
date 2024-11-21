import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def create_visualization(video_path, embedding_path, output_path=None):
    """Create a visualization with video on top and PCA embedding below"""
    
    # Load pre-computed embedding
    embedding = np.load(embedding_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup the figure
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((2, 1), (0, 0))  # Video
    ax2 = plt.subplot2grid((2, 1), (1, 0))  # PCA plot
    
    # Initialize video display
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = ax1.imshow(frame_rgb)
    ax1.axis('off')
    
    # Initialize PCA plot
    time = np.arange(len(embedding)) / fps
    line, = ax2.plot(time, embedding, 'b-', alpha=0.5)
    current_time_line = ax2.axvline(x=0, color='r')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('PCA Component 1')
    
    def update(frame_idx):
        # Update video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im.set_array(frame_rgb)
            
            # Update time indicator
            current_time = frame_idx / fps
            current_time_line.set_xdata([current_time, current_time])
            
        return im, current_time_line
    
    # Create animation
    anim = FuncAnimation(
        fig, update,
        frames=range(0, total_frames, 2),  # Skip every other frame for speed
        interval=1000/fps,  # in milliseconds
        blit=True
    )
    
    if output_path:
        anim.save(output_path, writer='ffmpeg')
    else:
        plt.show()
    
    cap.release()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize video with PCA embedding')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('embedding_path', help='Path to .npy file containing PCA embedding')
    parser.add_argument('--output', help='Path to output video (optional)')
    args = parser.parse_args()
    
    create_visualization(args.video_path, args.embedding_path, args.output)

if __name__ == '__main__':
    main()
