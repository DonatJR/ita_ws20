#!/bin/bash

### Script for data extraction. ###

GDIR="grobid"

# Switch to scraper running directory
echo "Data extraction begins..."
echo "Scraping pdfs..."
cd scraper
# Run spider to scrape and save pdfs to '../data/pdfs'
scrapy crawl jmlr
# Download Grobid if not already installed
if [ -d "$GDIR" ]; then
    echo "GROBID already installed..."
    echo "Proceed to setting up Grobid server..."
else
    echo "Grobid not installed.."
    echo "Cloning Grobid from https://github.com/kermitt2/grobid.git..."
    git clone https://github.com/kermitt2/grobid.git
    echo "Cloning Grobid python client from https://github.com/kermitt2/grobid_client_python..."
    git clone https://github.com/kermitt2/grobid_client_python
    
fi
echo "Proceed to setting up Grobid server..."
cd grobid
echo "Build Grobid with Gradle..."
{
      ./gradlew clean install
} || {
     echo "Failed to start the server. Please make sure that Grobid is installed in a path with no parent directories containing spaces."
 }
 ./gradlew clean install
echo "Start server with Gradle..."
./gradlew run &
GPID=$!
sleep 80s
cd ../grobid_client_python
# Send scraped pdfs to Grobid server
echo "Process PDFs with Grobid..."
python3 grobid_client.py --input ../../data/pdfs/ --output ../../data/tei/ processHeaderDocument

# Parse XML's created by Grobid
echo "Prase XMLs with Grobid..."
cd ..
python3 parser.py
echo "JSON file with parsed XMLs saved in data..."

echo "Close Grobid server..."
kill $GPID
