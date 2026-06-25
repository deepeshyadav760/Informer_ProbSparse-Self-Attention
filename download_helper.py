import os
import sys
import requests

def download_wheel(package_name, version=None):
    print(f"Querying PyPI for package '{package_name}'...")
    url = f"https://pypi.org/pypi/{package_name}/json"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    
    # Get releases
    releases = data.get("releases", {})
    if not version:
        version = data["info"]["version"]
    
    print(f"Target version: {version}")
    files = releases.get(version, [])
    
    # Look for cp311-cp311-win_amd64.whl or py3-none-any.whl
    target_file = None
    for f in files:
        filename = f["filename"]
        if filename.endswith(".whl"):
            # Check if it matches python 3.11 and windows amd64
            if "cp311" in filename and "win_amd64" in filename:
                target_file = f
                break
            elif "py3-none-any" in filename:
                # Fallback to pure python wheel if no architecture specific wheel
                target_file = f
    
    if not target_file and files:
        # Just grab the first wheel if no specific match
        for f in files:
            if f["filename"].endswith(".whl"):
                target_file = f
                break
                
    if not target_file:
        print(f"Error: No suitable wheel found for {package_name} {version}")
        sys.exit(1)
        
    download_url = target_file["url"]
    filename = target_file["filename"]
    print(f"Downloading {filename} from {download_url}...")
    
    os.makedirs("pip_tmp", exist_ok=True)
    dest_path = os.path.join("pip_tmp", filename)
    
    # Download file
    with requests.get(download_url, stream=True) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as out_f:
            for chunk in resp.iter_content(chunk_size=8192):
                out_f.write(chunk)
                
    print(f"[OK] Downloaded successfully: {dest_path}")
    return dest_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_helper.py <package_name> [version]")
        sys.exit(1)
    pkg = sys.argv[1]
    ver = sys.argv[2] if len(sys.argv) > 2 else None
    download_wheel(pkg, ver)
