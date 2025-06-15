#!/usr/bin/env python3
import os
import zipfile
from io import BytesIO
from urllib.request import urlopen, Request

ZIP_URL = "https://github.com/LUMII-AILab/FullStack/archive/refs/heads/master.zip"
PREFIX = "FullStack-master/NamedEntities/data/"
DEST_DIR = os.path.join("data", "LUMII-AiLab")

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Downloading {ZIP_URL} …")
    req = Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    archive = urlopen(req).read()

    print("Extracting NamedEntities/data …")
    with zipfile.ZipFile(BytesIO(archive)) as z:
        for member in z.namelist():
            if not member.startswith(PREFIX):
                continue
            rel_path = member[len(PREFIX):]
            target = os.path.join(DEST_DIR, rel_path)
            if member.endswith("/"):
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    print(f"Downloaded into {DEST_DIR}")

if __name__ == "__main__":
    main()
