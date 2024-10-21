import re

def sanitize_filename(filename: str) -> str:
    """
    Sanitize the input string to be a valid filename by removing or replacing invalid characters.
    
    Args:
        filename (str): The original filename string.
        
    Returns:
        str: A sanitized filename string.
    """
    # Define a regex pattern for invalid characters
    invalid_chars_pattern = r'[<>:"/\\|?*]'
    
    # Replace invalid characters with an underscore
    sanitized = re.sub(invalid_chars_pattern, '_', filename)
    
    # Remove any trailing periods or spaces
    sanitized = sanitized.strip().rstrip('.')
    
    # Optionally, limit the filename length (common max is 255 characters)
    max_length = 255
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized
