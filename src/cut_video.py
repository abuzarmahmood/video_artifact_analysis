#!/usr/bin/env python3
"""
Script to cut a video to a specified duration.
"""

from moviepy.editor import VideoFileClip
import argparse

def cut_video(input_path, output_path, duration):
    """
    Cut a video to specified duration in seconds.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to save the cut video
        duration (float): Desired duration in seconds
    """
    video = VideoFileClip(input_path)
    
    # If requested duration is longer than video, use full video
    final_duration = min(duration, video.duration)
    
    # Cut the video
    cut_video = video.subclip(0, final_duration)
    
    # Write the result
    cut_video.write_videofile(output_path)
    
    # Clean up
    video.close()
    cut_video.close()

def main():
    parser = argparse.ArgumentParser(description='Cut a video to specified duration')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', help='Path to save the cut video')
    parser.add_argument('duration', type=float, help='Desired duration in seconds')
    
    args = parser.parse_args()
    
    cut_video(args.input_video, args.output_video, args.duration)

if __name__ == "__main__":
    main()
