# /intelligent-query-retrieval-system/app/utils.py

import requests
import logging
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentDownloadError(Exception):
    """Custom error for download failures."""
    pass

def download_document(url: str, save_dir: str = "temp_docs") -> Path:
    """
    Downloads a document from a given URL and saves it locally.

    Args:
        url (str): URL of the document to download.
        save_dir (str): Directory where the document will be saved.

    Returns:
        Path: Path to the downloaded document.

    Raises:
        DocumentDownloadError: If the download fails.
        ValueError: If file name or extension is invalid.
    """
    try:
        # Ensure the directory exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Extract file name from URL
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name or "document"
        save_path = Path(save_dir) / filename

        # Validate the file extension
        if not save_path.suffix:
            raise ValueError("The downloaded file must have a valid extension (e.g. .pdf, .docx).")

        logger.info(f"Downloading document from {url} to {save_path}")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info(f"Document successfully downloaded to {save_path}")
        return save_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Download error from {url}: {e}")
        raise DocumentDownloadError(f"Failed to download document: {url}") from e