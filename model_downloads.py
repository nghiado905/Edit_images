"""Download selected public Hugging Face release files with resumable HTTPS."""

from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def curl_download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for resumable model downloads.")
    subprocess.run([
        curl, "--fail", "--location", "--silent", "--show-error", "--retry", "3",
        "--retry-delay", "2", "--connect-timeout", "20", "--speed-limit", "1024",
        "--speed-time", "120", "--continue-at", "-", "--output", str(destination), url,
    ], check=True)


def release_info(repo_id):
    """Use curl for metadata too; avoids slow Python HTTP connections on Windows."""
    cache = ROOT / ".cache" / "model-manifests"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / (repo_id.replace("/", "--") + ".json")
    url = f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}?blobs=true"
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        # Fetch metadata without append/resume: the model release may have changed.
        response = subprocess.run(
            [curl, "--fail", "--location", "--silent", "--show-error", "--retry", "2",
             "--connect-timeout", "20", "--max-time", "90", url],
            check=True, capture_output=True,
        ).stdout
    else:
        with urlopen(url, timeout=30) as response:
            response = response.read()
    info = json.loads(response)
    # This is a downloaded manifest, not executable source.
    destination.write_bytes(response)
    return info


def selected_release(repo_id, patterns):
    info = release_info(repo_id)
    files = [file for file in info["siblings"]
             if any(fnmatch.fnmatchcase(file["rfilename"], pattern) for pattern in patterns)]
    if not files:
        raise RuntimeError(f"No release files matched in {repo_id}: {patterns}")
    return {"repo_id": repo_id, "revision": info["sha"], "files": files}


def download_release(release, directory, workers=2):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    def download(file):
        relative = Path(file["rfilename"])
        destination = (directory / relative).resolve()
        if not destination.is_relative_to(directory):
            raise ValueError(f"Invalid release path: {relative}")
        lfs = file.get("lfs") or {}
        expected = file.get("size") or lfs.get("size")
        if destination.is_file() and expected is not None and destination.stat().st_size == expected:
            return
        partial = destination.with_name(destination.name + ".partial")
        print(f"Downloading {release['repo_id']}/{relative} ({(expected or 0) / 1024**3:.2f} GiB)", flush=True)
        url = (f"https://huggingface.co/{release['repo_id']}/resolve/{release['revision']}/"
               f"{quote(file['rfilename'], safe='/')}")
        if not partial.is_file() or expected is None or partial.stat().st_size < expected:
            curl_download(url, partial)
        if expected is not None and partial.stat().st_size != expected:
            raise RuntimeError(f"Incomplete download: {partial}. Rerun setup to resume.")
        expected_hash = lfs.get("sha256")
        if expected_hash:
            with partial.open("rb") as downloaded:
                digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
            if digest != expected_hash:
                raise RuntimeError(f"SHA-256 mismatch: {partial}. Remove this partial file before retrying.")
        partial.replace(destination)
        print(f"Ready: {relative}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(download, release["files"]))
    manifest = {key: value for key, value in release.items() if key != "files"}
    manifest["files"] = [{"path": f["rfilename"], "size": f.get("size")} for f in release["files"]]
    (directory / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
