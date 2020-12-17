import os
from pathlib import Path
from typing import Optional, TypeVar

import requests
from tqdm import tqdm

Response = TypeVar('Response', bound=requests.models.Response)


def _download_file_from_google_drive(id: str, destination: str) -> None:
    """Download the data from the Google drive public URL.
    This method will create a session instance to persist the requests and reuse TCP connection for the large files.
    Args:
        id: File ID of Google drive URL.
        destination: Destination path where the data needs to be stored.
    """
    URL = "https://drive.google.com/uc?export=download"
    CHUNK_SIZE = 128
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    progress = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                f.write(chunk)
    progress.close()


def _get_confirm_token(response: Response) -> str:
    """Retrieve the token from the cookie jar of HTTP request to keep the session alive.
    Args:
        response: Response object of the HTTP request.
    Returns"
        The value of cookie in the response object.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def download_yolo_weights(root_dir: Optional[str] = None) -> str:
    """Download YOLOv4 weights and return the path.
    Sourced from https://drive.google.com/uc?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT&export=download. This method will
        download the data to local storage if the data has not been previously downloaded.
    Args:
        root_dir: The path to store the downloaded data. When `path` is not provided, the data will be saved into
            `fastestimator_data` under the user's home directory.
    Returns:
        train_data
    """
    home = str(Path.home())

    if root_dir is None:
        root_dir = os.path.join(home, 'fastestimator_data', 'YOLO')
    else:
        root_dir = os.path.join(os.path.abspath(root_dir), 'YOLO')
    os.makedirs(root_dir, exist_ok=True)

    yolo_weights_path = os.path.join(root_dir, 'yolov4.weights')

    if not os.path.exists(yolo_weights_path):
        # download
        print("Downloading data to {}".format(root_dir))
        _download_file_from_google_drive('1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT', yolo_weights_path)

    return yolo_weights_path
