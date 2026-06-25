import os
import sys
import subprocess
import requests

PACKAGES = ["seaborn", "streamlit", "mlflow", "plotly", "tqdm", "earthengine-api"]

def download_wheel(package_name):
    print(f"Querying PyPI for package '{package_name}'...")
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Failed to query PyPI for {package_name}: {e}")
        return None
        
    version = data["info"]["version"]
    print(f"Target version for {package_name}: {version}")
    releases = data.get("releases", {})
    files = releases.get(version, [])
    
    target_file = None
    # 1. Look for cp311-cp311-win_amd64.whl
    for f in files:
        filename = f["filename"]
        if filename.endswith(".whl") and "cp311" in filename and "win_amd64" in filename:
            target_file = f
            break
            
    # 2. Look for any cp311 wheel
    if not target_file:
        for f in files:
            filename = f["filename"]
            if filename.endswith(".whl") and "cp311" in filename:
                target_file = f
                break
                
    # 3. Look for py3-none-any.whl or py2.py3-none-any.whl
    if not target_file:
        for f in files:
            filename = f["filename"]
            if filename.endswith(".whl") and "none-any" in filename:
                target_file = f
                break
                
    # 4. Look for any wheel
    if not target_file:
        for f in files:
            filename = f["filename"]
            if filename.endswith(".whl"):
                target_file = f
                break
                
    if not target_file:
        print(f"Error: No suitable wheel found for {package_name} {version}")
        return None
        
    download_url = target_file["url"]
    filename = target_file["filename"]
    print(f"Downloading {filename}...")
    
    os.makedirs("pip_tmp", exist_ok=True)
    dest_path = os.path.join("pip_tmp", filename)
    
    # Download file
    try:
        with requests.get(download_url, stream=True) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as out_f:
                for chunk in resp.iter_content(chunk_size=8192):
                    out_f.write(chunk)
        print(f"Downloaded successfully to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Download failed for {package_name}: {e}")
        return None

def main():
    downloaded_wheels = []
    for pkg in PACKAGES:
        # Check if already installed
        try:
            # We map some packages to their import names if different
            import_name = pkg
            if pkg == "earthengine-api":
                import_name = "ee"
            __import__(import_name)
            print(f"Package '{pkg}' is already installed. Skipping.")
        except ImportError:
            wheel_path = download_wheel(pkg)
            if wheel_path:
                downloaded_wheels.append(wheel_path)
                
    if not downloaded_wheels:
        print("No packages need to be installed.")
        return
        
    print("\nInstalling downloaded wheels...")
    for wheel in downloaded_wheels:
        print(f"Installing {wheel}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", wheel], check=True)
            print(f"Successfully installed {wheel}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {wheel}: {e}")
            
if __name__ == "__main__":
    main()
