import os
import gdown

def download_model(url):
    MODEL_PATH = "poetry_generator.pth"
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        # Convert Google Drive sharing URL to direct download URL
        file_id = url.split('/')[5]
        direct_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(direct_url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")
    else:
        print("Model file already exists.")

if __name__ == "__main__":
    # Default URL in case no secret is provided
    default_url = "https://drive.google.com/file/d/14xR_Q2V7QsitWP2kVpCqMS3hS399GHpm/view?usp=drive_link"
    download_model(default_url)
