"""
Utility functions
"""
import os
import shutil
from pathlib import Path


def clean_output_folder(output_dir: Path) -> None:
    """
    Clean output folder by removing all contents
    
    Args:
        output_dir: Path to output directory
    """
    if output_dir.exists():
        for item in output_dir.glob('*'):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"Folder {output_dir} cleaned")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Folder {output_dir} created")


def get_image_files(directory: Path, extensions: set = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}) -> list:
    """
    Get all image files in a directory
    
    Args:
        directory: Directory to search
        extensions: Set of image extensions
        
    Returns:
        List of image file paths
    """
    return [f for f in directory.iterdir() if f.suffix.lower() in extensions]