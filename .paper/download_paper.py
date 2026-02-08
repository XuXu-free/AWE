import requests
import tarfile
import os
import shutil
import urllib3

def download_and_extract_paper(arxiv_id, output_dir):
    url = f"https://arxiv.org/src/{arxiv_id}"
    tar_path = os.path.join(output_dir, f"{arxiv_id}.tar.gz")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading {url}...")
    # Disable SSL verification
    urllib3.disable_warnings()
    response = requests.get(url, stream=True, verify=False)
    if response.status_code == 200:
        with open(tar_path, 'wb') as f:
            f.write(response.raw.read())
        print("Download complete.")
        
        print("Extracting...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=output_dir)
            print("Extraction complete.")
        except tarfile.ReadError:
            print("Error: Not a valid tar file. It might be a direct PDF or other format.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_and_extract_paper("2501.14576", "d:\\Projects\\AWE\\.paper")
