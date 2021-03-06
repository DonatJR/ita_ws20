"""
This is a parser for the XMLs created by Grobid.
It creates the JSON file where we save our data.
"""

import datetime
import json
import os
import xml.etree.ElementTree as ET
from math import nan
from pathlib import Path

input_path = Path("../data/tei/")
keys = [
    "title",
    "abstract",
    "keywords",
    "author",
    "ref",
    "datasource",
    "datasource_url",
]
datasource = "Journal of Machine Learning Research"
datasource_url = "https://jmlr.csail.mit.edu"
papers = []


def read_xmls():
    """ Loading the XML files """
    for filename in os.listdir(input_path):
        if not filename.endswith(".xml"):
            continue

        fullname = os.path.join(input_path, filename)
        try:
            tree = ET.parse(fullname)
        except Exception:
            print("could not parse {}\n".format(filename))
        parse_xml(tree.getroot(), filename)

    create_data_file()


def parse_xml(root, filename):
    """ Parsing the XML files to extract information """
    global keys
    global datasource
    global datasource_url

    print("curently parsing {}\n".format(filename))

    title = ""
    authors = []
    keywords = []
    abstract = []

    for child in root.iter():
        tag = child.tag.split("}")[1]

        if tag == "title" and title == "":
            title = child.text
        if tag == "persName":
            if len([subtag for subtag in child]) >= 2:
                names = [subtag.text for subtag in child]
                author_name = ""
                for name in names:
                    author_name = author_name + " " + name

                authors.append(author_name.strip())
        if tag == "keywords":
            keywords = [term.text for term in child]
        if tag == "abstract":
            abstract = [p.text for p in child]

    # Ref will be the name of the file
    filename = (
        filename.replace(";", r"/")
        .replace("tei.xml", "pdf")
        .replace("https///", "https://")
    )

    if abstract == []:
        abstract = nan
    else:
        abstract = abstract[0]

    vals = [
        title,
        abstract,
        keywords,
        authors,
        filename,
        datasource,
        datasource_url,
    ]

    papers.append(dict(zip(keys, vals)))


def create_data_file():
    """ Create and save JSON with extracted data """
    with open("../data/data.json", "w") as f:
        json.dump({"papers": papers}, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    read_xmls()
