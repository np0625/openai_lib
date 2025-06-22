def read_file_to_string(file_path: str) -> str:
    """
    Read a UTF-8 encoded file and return its contents as a string.

    Args:
        file_path (str): Path to the file to read

    Returns:
        str: Contents of the file as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
        UnicodeDecodeError: If the file is not valid UTF-8
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"File {file_path} is not valid UTF-8")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")