import re
import json
from typing import List, Dict, Any
from pathlib import Path


def parse_time_to_ms(time_str: str) -> int:
    """
    Convert SRT time format (HH:MM:SS,mmm) to milliseconds.
    
    Args:
        time_str (str): Time string in format "HH:MM:SS,mmm"
        
    Returns:
        int: Time in milliseconds
    """
    # Parse hours, minutes, seconds, milliseconds
    hours, minutes, seconds_ms = time_str.split(':')
    seconds, milliseconds = seconds_ms.split(',')
    
    total_ms = (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)
    return total_ms


def parse_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parse SRT subtitle file and convert to JSON format.
    
    Args:
        srt_path (str): Path to the SRT file
        
    Returns:
        List[Dict[str, Any]]: List of subtitle entries in JSON format
    """
    with open(srt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split content into individual subtitle blocks
    subtitle_blocks = re.split(r'\n\s*\n', content.strip())
    result = []
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue  # Skip invalid blocks
        
        try:
            # Parse index
            index = int(lines[0])
            
            # Parse timestamp line
            timestamp_line = lines[1]
            start_time, end_time = timestamp_line.split(' --> ')
            
            # Convert to milliseconds
            start_ms = parse_time_to_ms(start_time.strip())
            end_ms = parse_time_to_ms(end_time.strip())
            duration = end_ms - start_ms
            
            # Parse text content (remaining lines)
            text_lines = lines[2:]
            
            # Handle text array - split by new lines
            text_array = [line.strip() for line in text_lines if line.strip()]
            
            # Create the entry
            entry = {
                "text": text_array,
                "index": index,
                "start": start_ms,
                "end": end_ms,
                "duration": duration,
                "gender": "male"
            }
            
            result.append(entry)
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping invalid subtitle block: {block}")
            continue
    
    return result


def srt_to_json(srt_path: str, output_path: str = None) -> str:
    """
    Convert SRT file to JSON format and optionally save to file.
    
    Args:
        srt_path (str): Path to the SRT file
        output_path (str, optional): Path to save the JSON output. If None, returns JSON string
        
    Returns:
        str: JSON string if output_path is None, otherwise None
    """
    # Parse the SRT file
    subtitles = parse_srt_file(srt_path)
    
    # Convert to JSON
    json_output = json.dumps(subtitles, ensure_ascii=False, indent=4)
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(json_output)
        print(f"JSON output saved to: {output_path}")
        return None
    else:
        return json_output


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert SRT subtitle file to JSON format')
    parser.add_argument('srt_file', help='Path to the SRT file')
    parser.add_argument('-o', '--output', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Convert the file
    result = srt_to_json(args.srt_file, args.output)
    
    if result:
        print(result)


if __name__ == "__main__":
    main()
