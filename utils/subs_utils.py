from pathlib import Path
from pydub import AudioSegment
import subprocess
import json
import os
import tqdm

def ensure_folder_exists(folder_path):
    """Ensure that a folder exists at the specified path, creating it if necessary.
    
    Args:
        folder_path (str or Path): Path to the folder that should exist
    
    Returns:
        None: Prints status message about folder creation/existence
    
    Note:
        Creates parent directories if needed (mkdir -p equivalent)
    """
    if not Path(folder_path).exists():
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def change_speed(audio_file_path, save_file_path, file_name, start, end, min_speed, max_speed, sample_rate):
    """Adjust the speed of an audio file to match a target duration.
    
    Args:
        audio_file_path (str): Path to the input audio file
        save_file_path (str): Directory to save the output file
        file_name (str): Name of the output file
        start (int): Start time in milliseconds
        end (int): End time in milliseconds
        min_speed (float): Minimum allowed speed adjustment factor
        max_speed (float): Maximum allowed speed adjustment factor
        sample_rate (int): Target sample rate for output file
    
    Returns:
        None: Saves the adjusted audio file and prints status messages
    """
    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)
    
    # Calculate the speed adjustment
    current_duration = len(audio) / 1000  # Convert to seconds
    target_duration = (end - start) / 1000  # Convert to seconds
    speed_adjustment = current_duration / target_duration

    if speed_adjustment < min_speed:
        speed_adjustment = min_speed
    elif speed_adjustment > max_speed:
        speed_adjustment = max_speed
    
    print(f"Speed adjustment factor: {speed_adjustment}")

    # Prepare the FFmpeg command
    input_file = audio_file_path
    output_file = f"{save_file_path}/{file_name}"
    
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_file,
        "-filter:a", f"atempo={speed_adjustment}",
        "-ar", f'{sample_rate}',  # Set the output sample rate to match the input
        "-acodec", "pcm_s16le",  # Use 16-bit PCM codec for WAV
        "-y",  # Overwrite output file if it exists
        output_file
    ]

    # Execute the FFmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE)
        print(f"Speed adjusted audio saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")

def create_silence_audio(video_path, output_audio_path):
    """Create a silent audio file matching the duration of a video.
    
    Args:
        video_path (str): Path to the input video file
        output_audio_path (str): Path to save the output silent audio file
    
    Returns:
        None: Creates the silent audio file and prints status messages
    """
    # Get video duration using FFprobe
    probe_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    # Extract duration from the JSON output
    duration = float(data['format']['duration'])
    
    # Create silent audio using FFmpeg
    create_cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", str(duration),
        "-q:a", "0",
        "-map", "0",
        output_audio_path
    ]
    
    subprocess.run(create_cmd, check=True)
    
    print(f"Silent audio file created: {output_audio_path}")
    print(f"Duration: {duration} seconds")


def overlay_audio(silence_file, audio_files_folder, json_file_path, output_path):
    """Overlay multiple audio segments onto a silent base audio track.
    
    Args:
        silence_file (str): Path to the silent base audio file
        audio_files_folder (str): Directory containing audio segments to overlay
        json_file_path (str): Path to JSON file with timing information
        output_path (str): Path to save the final mixed audio file
    
    Returns:
        None: Creates the mixed audio file and prints status messages
    """
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)

    # Paths setup
    # target_audio_path = f'{config_data["modifiedFilesFolder"]}/{config_data["silenceFileName"]}.wav'  # Target audio file path
    # audio_files_folder = f'{config_data["modifiedFilesFolder"]}'  # Folder containing (index).wav files
    # json_file_path = f'{config_data["subsPath"]}'  # Path to your JSON file
    # output_path = f'{config_data["modifiedFilesFolder"]}/output.mp3'  # Output file path

    # Load the target audio file
    print("Loading target audio file...")
    target_audio = AudioSegment.from_file(silence_file)

    # Load and parse the JSON file
    print("Loading and parsing JSON data...")
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        entries = json.load(json_file)

    # Process each entry with a progress bar
    print("Processing audio overlays...")
    for entry in tqdm(entries, desc="Applying overlays"):
        index = entry['index']
        start = entry['start']
        end = entry['end']

        audio_file_path = os.path.join(audio_files_folder, f"{index}.wav")
        if os.path.exists(audio_file_path):
            audio_file = AudioSegment.from_file(audio_file_path)
            target_audio = target_audio.overlay(audio_file, position=start)

    # Export the combined audio to a new file
    print("Exporting the combined audio...")
    target_audio.export(output_path, format="mp3")
    print("Done.")
