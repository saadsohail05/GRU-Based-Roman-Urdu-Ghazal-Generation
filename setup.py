import os
import requests
from tqdm import tqdm

def download_model():
    # Replace this URL with your actual model hosting URL (e.g., from Google Drive, Dropbox, etc.)
    MODEL_URL = "https://drive.google.com/file/d/14xR_Q2V7QsitWP2kVpCqMS3hS399GHpm/view?usp=drive_link"
    MODEL_PATH = "poetry_generator.pth"

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(MODEL_PATH, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

if __name__ == "__main__":
    download_model()
