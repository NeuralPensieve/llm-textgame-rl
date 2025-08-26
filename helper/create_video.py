import cv2
import os
import glob
import argparse
from pathlib import Path

def create_video_from_pngs(input_folder, output_video, fps=1, video_codec='mp4v', crop_top_percent=100):
    """
    Create a video from PNG files in a folder.
    
    Args:
        input_folder (str): Path to folder containing PNG files
        output_video (str): Output video file path
        fps (float): Frames per second for the output video
        video_codec (str): Video codec to use ('mp4v', 'XVID', etc.)
        crop_top_percent (float): Percentage of image height to keep from top (1-100)
    """
    
    # Get all PNG files and sort by name
    png_pattern = os.path.join(input_folder, "*.png")
    image_files = sorted(glob.glob(png_pattern))
    
    if not image_files:
        print(f"No PNG files found in {input_folder}")
        return False
    
    print(f"Found {len(image_files)} PNG files")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Could not read the first image: {image_files[0]}")
        return False
    
    original_height, width, channels = first_image.shape
    
    # Calculate crop parameters
    if crop_top_percent <= 0 or crop_top_percent > 100:
        print(f"Error: crop_top_percent must be between 1 and 100, got {crop_top_percent}")
        return False
    
    height = int(original_height * crop_top_percent / 100)
    
    print(f"Original dimensions: {width}x{original_height}")
    if crop_top_percent < 100:
        print(f"Keeping top {crop_top_percent}% of image ({height} pixels)")
        print(f"Final video dimensions: {width}x{height}")
    else:
        print(f"Video dimensions: {width}x{height}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return False
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue
        
        # Apply cropping if specified
        if crop_top_percent < 100:
            img_height = img.shape[0]
            keep_height = int(img_height * crop_top_percent / 100)
            img = img[0:keep_height, :, :]  # Keep only top portion
        
        # Resize image if dimensions don't match the target dimensions
        if img.shape[:2] != (height, width):
            print(f"Resizing {os.path.basename(image_path)} from {img.shape[1]}x{img.shape[0]} to {width}x{height}")
            img = cv2.resize(img, (width, height))
        
        # Write frame to video
        video_writer.write(img)
    
    # Release everything
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video created successfully: {output_video}")
    print(f"Video settings: {fps} fps, {len(image_files)} frames, duration: {len(image_files)/fps:.2f} seconds")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create video from PNG files in a folder")
    parser.add_argument("input_folder", help="Path to folder containing PNG files")
    parser.add_argument("-o", "--output", default="output_video.mp4", help="Output video filename (default: output_video.mp4)")
    parser.add_argument("-f", "--fps", type=float, default=1.0, help="Frames per second (default: 1.0)")
    parser.add_argument("-c", "--codec", default="mp4v", help="Video codec (default: mp4v)")
    parser.add_argument("--crop-top", type=float, default=100, help="Percentage of image height to keep from top (1-100, default: 100 = keep full image)")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the video
    success = create_video_from_pngs(
        input_folder=args.input_folder,
        output_video=args.output,
        fps=args.fps,
        video_codec=args.codec,
        crop_top_percent=args.crop_top
    )
    
    if not success:
        print("Failed to create video")

# Example usage as a module
def create_video_simple(folder_path, output_path="output.mp4", fps=1, crop_top_percent=100):
    """
    Simplified function for direct use in other scripts.
    
    Args:
        folder_path (str): Path to folder with PNG files
        output_path (str): Output video file path
        fps (float): Frames per second
        crop_top_percent (float): Percentage of image height to keep from top (1-100)
    
    Returns:
        bool: True if successful, False otherwise
    """
    return create_video_from_pngs(folder_path, output_path, fps, crop_top_percent=crop_top_percent)

if __name__ == "__main__":
    main()