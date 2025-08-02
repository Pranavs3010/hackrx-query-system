# /intelligent-query-retrieval-system/app/utils.py

import requests
import logging
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_document(url: str, save_dir: str = "temp_docs") -> Path:
    """
    Downloads a document from a URL, preserving its file extension.

    Args:
        url (str): The URL of the document.
        save_dir (str): The local directory to save the file.

    Returns:
        Path: The local path to the downloaded document.
    """
    try:
        # Ensure the save directory exists
        Path(save_dir).mkdir(exist_ok=True)

        # Get a clean filename from the URL path
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        if not filename:
            filename = "document" # Fallback filename

        save_path = Path(save_dir) / filename
        
        logger.info(f"Downloading document from {url} to {save_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Will raise an HTTPError for bad responses

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded document.")
        return save_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file from {url}: {e}")
        raise