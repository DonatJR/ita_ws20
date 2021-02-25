#!/bin/bash

### Script for data extraction. ###

GDIR="grobid"

echo "Data extraction begins..."
echo "Scraping pdfs..."
# Switch to scraper running directory
cd scraper
# Run spider to scrape and save pdfs to '../../data/pdfs'
scrapy crawl jmlr
cd ..
Download Grobid if not already installed
if [ -d "$GDIR" ]; then
    echo "Grobid already installed..."
else
    echo "Grobid not installed..."
    echo "Cloning Grobid from https://github.com/kermitt2/grobid.git..."
    git clone https://github.com/kermitt2/grobid.git
    echo "Cloning Grobid python client from https://github.com/kermitt2/grobid_client_python..."
    git clone https://github.com/kermitt2/grobid_client_python
    
fi
echo "Proceed to setting up Grobid server..."
cd grobid
echo "Building Grobid with Gradle..."
{
      ./gradlew clean install
} || {
     echo "Failed to start the server. Please make sure that Grobid is installed in a path with no parent directories containing spaces."
}
./gradlew clean install
echo "Starting Grobid server with Gradle..."
./gradlew run &
GPID=$!
sleep 80s
cd ../grobid_client_python
# Send scraped pdfs to Grobid server
echo "Processing PDFs with Grobid..."
python3 grobid_client.py --input ../../data/pdfs/ --output ../../data/tei/ processHeaderDocument
# Parse XML's created by Grobid
echo "Parsing XMLs with Grobid..."
cd ..
python3 parser.py
echo "Saving JSON file with parsed XMLs in 'data'..."
echo "Closing Grobid server..."
kill $GPID
