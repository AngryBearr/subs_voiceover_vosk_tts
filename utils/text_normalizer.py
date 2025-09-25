import sys
import json
import ast
from ruaccent import RUAccent
from runorm import RUNorm

cust_dict = {'аганоа': 'аган+оа', 'санаапу': 'сан+аапу', 'вавау': 'вав+ау', 'дэз': 'д+эз', 'кюча': 'к+юча', 'огакор': 'ог+акор' }

accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, custom_dict=cust_dict, device="CUDA")
normalizer = RUNorm()
normalizer.load(workdir="./runorm_cache", model_size="big", device="cuda")

# Default batch size for subtitle processing
DEFAULT_BATCH_SIZE = 10

def replace_ellipsis(text):
    return text.replace('...', '..')

def restore_ellipsis(text):
    return text.replace('..', '...')

# text = 'На двери висит замок. Ежик нашел в лесу ягоды. Эти 10 ягод он ежик оставил себе. Зачем ему ягоды?'
# print(accentizer.process_all(text))

def transform_json(obj, batch_size=DEFAULT_BATCH_SIZE):
    # First, normalize all subtitles individually
    normalized_items = []
    for item in obj:
        if 'text' in item and isinstance(item['text'], list):
            # If text is a list, concatenate it into a string
            if len(item['text']) > 1:
                joined_text = ' '.join(item['text'])
            else:
                joined_text = item['text'][0] if item['text'] else ""
            
            # Apply initial normalization
            no_elipsis_text = replace_ellipsis(joined_text)
            normalized_text = normalizer.norm(no_elipsis_text)
            
            # Store normalized text with original item
            normalized_item = item.copy()
            normalized_item['normalized_text'] = normalized_text
            normalized_items.append(normalized_item)
    
    # Process in batches
    for i in range(0, len(normalized_items), batch_size):
        batch = normalized_items[i:i+batch_size]
        
        # Extract normalized texts as an array of strings
        batch_texts = [item['normalized_text'] for item in batch]
        
        # Convert array to string representation with brackets preserved
        batch_string = str(batch_texts)
        print(batch_string)
        # Process the entire batch string with ruaccent, skipping array syntax
        processed_batch_string = accentizer.process_all(batch_string, skip_regex='[\[\]]')
        print(processed_batch_string)
        # Parse the result back to a list
        try:
            processed_texts = ast.literal_eval(processed_batch_string)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing processed batch: {e}")
            # Fallback: process each item individually if batch processing fails
            processed_texts = [accentizer.process_all(text) for text in batch_texts]
        
        # Update items with processed texts
        for j, processed_text in enumerate(processed_texts):
            if j < len(batch):
                batch[j]['text'] = [restore_ellipsis(processed_text)]
    
    # Create the final transformed JSON data
    transformed_json_data = []
    for item in normalized_items:
        if 'normalized_text' in item:
            # Remove the temporary normalized_text field
            new_item = item.copy()
            del new_item['normalized_text']
            transformed_json_data.append(new_item)
    
    return transformed_json_data

def save_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def normalize_text_file(input_path, output_path, output_format='json', batch_size=DEFAULT_BATCH_SIZE):
    """
    Normalize the text in a JSON file and save/output in the specified format.
    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to save the output.
        output_format (str): 'json' (default) or 'txt'.
        batch_size (int): Number of subtitles to process in each batch.
    """
    with open(input_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    transformed_data = transform_json(data, batch_size=batch_size)

    if output_format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=2)
    elif output_format == 'txt':
        # Flatten all transformed lines into a single list
        lines = []
        for item in transformed_data:
            if 'text' in item and isinstance(item['text'], list):
                lines.extend(item['text'])
        txt_content = '\n'.join(lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
    else:
        raise ValueError("output_format must be 'json' or 'txt'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        batch_size = DEFAULT_BATCH_SIZE
        
        # Check if batch size is provided as third argument
        if len(sys.argv) > 3 and sys.argv[3].isdigit():
            batch_size = int(sys.argv[3])

        try:
            with open(json_file_path, 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
                print(f"Loaded {len(data)} subtitle entries")

            transformed_data = transform_json(data, batch_size=batch_size)

            if len(sys.argv) > 2:
                # If a second argument is provided, save the transformed JSON to this path
                output_file_path = sys.argv[2]
                save_to_file(transformed_data, output_file_path)
                print(f"Transformed data saved to {output_file_path}")
            else:
                # If no second argument, print the transformed JSON
                print(json.dumps(transformed_data, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Usage: python text_normalizer.py <input_json_file> [output_json_file] [batch_size]")
