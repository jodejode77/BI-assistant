#!/bin/bash

set -e
set -o pipefail

COMPETITION="home-credit-default-risk"
DATA_DIR="./data"

trap 'echo ""; echo "Download interrupted"; exit 1' INT TERM

mkdir -p "$DATA_DIR"

echo "Starting download of competition: $COMPETITION"
echo "Download directory: $DATA_DIR"
echo ""

echo "Downloading files (this may take a while)..."
echo ""

kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR" --force &
DOWNLOAD_PID=$!

while kill -0 $DOWNLOAD_PID 2>/dev/null; do
    if [ -d "$DATA_DIR" ]; then
        TOTAL_SIZE=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        FILE_COUNT=$(find "$DATA_DIR" -type f 2>/dev/null | wc -l)
        echo -ne "\rDownloading... Size: $TOTAL_SIZE, Files: $FILE_COUNT    "
    fi
    sleep 2
done

wait $DOWNLOAD_PID
DOWNLOAD_STATUS=$?
echo ""

if [ $DOWNLOAD_STATUS -eq 0 ]; then
    DOWNLOAD_SUCCESS=true
else
    DOWNLOAD_SUCCESS=false
fi

if [ "$DOWNLOAD_SUCCESS" = true ]; then
    echo ""
    echo "Download completed. Extracting files..."
    cd "$DATA_DIR"
    if ls *.zip 1> /dev/null 2>&1; then
        for zipfile in *.zip; do
            echo "Extracting: $zipfile"
            unzip -o "$zipfile"
        done
        echo "Cleaning up zip files..."
        rm -f *.zip
    fi
    echo ""
    echo "Dataset downloaded and extracted successfully to $DATA_DIR"
else
    echo ""
    echo "Failed to download dataset"
    exit 1
fi

