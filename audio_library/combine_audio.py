import os
from pydub import AudioSegment
import glob

def combine_aif_files(input_folder, output_filename="combined_audio.aif"):
    """
    Combines multiple .aif audio files from a folder into a single file.
    
    Args:
        input_folder (str): Path to folder containing .aif files
        output_filename (str): Name for the output file
    """
    # Get all .aif files in the folder
    aif_files = glob.glob(os.path.join(input_folder, "*.aif"))
    
    if not aif_files:
        print("No .aif files found in the specified folder!")
        return
    
    # Sort files alphabetically (optional - you can remove this if order doesn't matter)
    aif_files.sort()
    
    print(f"Found {len(aif_files)} .aif files to combine:")
    for file in aif_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load the first audio file
    combined = AudioSegment.from_file(aif_files[0], format="aiff")
    
    # Append all other audio files
    for aif_file in aif_files[1:]:
        audio = AudioSegment.from_file(aif_file, format="aiff")
        combined += audio
    
    # Calculate total duration
    duration_seconds = len(combined) / 1000.0
    print(f"\nTotal duration: {duration_seconds:.2f} seconds")
    
    # Export the combined audio
    output_path = os.path.join(input_folder, output_filename)
    combined.export(output_path, format="aiff")
    print(f"Combined audio saved to: {output_path}")
    
    return output_path

def combine_with_silence(input_folder, silence_ms=500, output_filename="combined_with_gaps.aif"):
    """
    Combines .aif files with silence gaps between them.
    
    Args:
        input_folder (str): Path to folder containing .aif files
        silence_ms (int): Milliseconds of silence between clips
        output_filename (str): Name for the output file
    """
    aif_files = glob.glob(os.path.join(input_folder, "*.aif"))
    
    if not aif_files:
        print("No .aif files found!")
        return
    
    aif_files.sort()
    
    # Create silence segment
    silence = AudioSegment.silent(duration=silence_ms)
    
    # Start with the first file
    combined = AudioSegment.from_file(aif_files[0], format="aiff")
    
    # Add remaining files with silence between
    for aif_file in aif_files[1:]:
        combined += silence
        audio = AudioSegment.from_file(aif_file, format="aiff")
        combined += audio
    
    # Save the result
    output_path = os.path.join(input_folder, output_filename)
    combined.export(output_path, format="aiff")
    print(f"Combined audio with gaps saved to: {output_path}")
    
    return output_path

# Example usage:
if __name__ == "__main__":
    # Specify the folder containing your .aif files
    folder_path = "files"
    
    # Method 1: Combine files directly (no gaps)
    #combine_aif_files(folder_path, "jarvis_combined.aif")
    
    # Method 2: Combine with small silence gaps (can help with voice cloning)
    combine_with_silence(folder_path, silence_ms=200, output_filename="jarvis_with_gaps.aif")