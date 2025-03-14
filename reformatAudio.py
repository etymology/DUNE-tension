import os
import glob
import numpy as np

def load_and_save_npz(input_file: str, output_file: str) -> None:
    """
    Loads an .npz file, extracts its audio data (assumed to be the first array),
    and saves a new .npz file with the audio data under the key "audio" and 
    a fixed sample_rate of 44100.
    
    Parameters:
        input_file (str): Path to the input .npz file.
        output_file (str): Path where the reformatted .npz file will be saved.
    """
    # Load the .npz file (returns a dict-like object)
    data = np.load(input_file)
    
    # Extract the first array found in the file as the audio data.
    audio_data = None
    for key in data.files:
        audio_data = data[key]
        break  # Use the first array encountered

    if audio_data is None:
        raise ValueError(f"No audio data found in file: {input_file}")

    # Save the reformatted data with the key "audio" and sample_rate=44100.
    np.savez(output_file, audio=audio_data, sample_rate=44100)

def main():
    # Define input and output directories
    input_folder = "audio"
    output_folder = "audio_formatted"
    
    # Create the output folder if it doesn't already exist.
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all .npz files in the input folder.
    npz_files = glob.glob(os.path.join(input_folder, "*.npz"))
    
    if not npz_files:
        print("No .npz files found in the 'audio' folder.")
        return
    
    # Process each .npz file.
    for input_path in npz_files:
        # Build the output file path with the same base name.
        base_name = os.path.basename(input_path)
        output_path = os.path.join(output_folder, base_name)
        
        print(f"Processing: {input_path} -> {output_path}")
        try:
            load_and_save_npz(input_path, output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    main()
