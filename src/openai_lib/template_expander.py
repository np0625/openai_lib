import yaml
import os
import sys
import argparse

def _read_file_contents(file_path, base_dir=None):
    """Helper function to read file contents, handling relative paths."""
    if base_dir:
        file_path = os.path.join(base_dir, file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

class FileLoader(yaml.SafeLoader):
    """Custom YAML loader that handles !file tags"""
    pass

def _file_constructor(loader, node):
    """Constructor for handling !file tags"""
    file_path = loader.construct_scalar(node)
    # Get base_dir from the loader instance
    base_dir = loader.base_dir if hasattr(loader, 'base_dir') else None
    return _read_file_contents(file_path, base_dir)

# Register the custom constructor with our FileLoader class
FileLoader.add_constructor('!file', _file_constructor)

def expand_yaml_template(yaml_path, required_fields=('instructions', 'data')):
    """
    Expands a YAML template file that may contain !file references.
    Returns a dict with all fields preserving their original data types.

    Args:
        yaml_path (str): Path to the YAML template file
        required_fields (tuple): Immutable sequence of required field names that must be present. Defaults to ('instructions', 'data')

    Returns:
        dict: Contains all fields with their original data types preserved. !file annotations are loaded as strings.
    """
    # Get the directory of the YAML file to resolve relative paths
    base_dir = os.path.dirname(os.path.abspath(yaml_path))

    # Create a loader instance and set its base directory
    loader = FileLoader(open(yaml_path, 'r', encoding='utf-8'))
    loader.base_dir = base_dir

    try:
        template = loader.get_single_data()
    finally:
        loader.dispose()

    # Ensure all required fields are present
    all_fields = set(template.keys())
    missing_fields = set(required_fields) - all_fields
    if missing_fields:
        raise ValueError(f"Missing required fields in YAML: {missing_fields}")

    # Return all fields with their original data types preserved
    # The !file constructor already handles converting file references to strings
    return template


def _parse_required_fields(arg_value: str | None):
    if not arg_value:
        return None
    fields = [field.strip() for field in arg_value.split(',') if field.strip()]
    return tuple(fields)


def main():
    parser = argparse.ArgumentParser(
        description="Expand a YAML template (supports !file) and print YAML to stdout"
    )
    parser.add_argument(
        "yaml_path",
        help="Path to the YAML template file",
    )
    parser.add_argument(
        "-r",
        "--required-fields",
        dest="required_fields",
        help=(
            "Comma-separated list of required field names. "
            "If omitted, defaults to ('instructions','data')."
        ),
    )

    args = parser.parse_args()

    required_fields = _parse_required_fields(args.required_fields)

    try:
        if required_fields is None:
            result = expand_yaml_template(args.yaml_path)
        else:
            result = expand_yaml_template(args.yaml_path, required_fields)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print YAML text to stdout
    yaml_text = yaml.safe_dump(result, sort_keys=False, allow_unicode=True)
    print(yaml_text, end="")


if __name__ == "__main__":
    main()
