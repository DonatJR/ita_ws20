#!/usr/bin/env python
import re
import io
import json
import requests
import pdftotext
from pathlib import Path
from bs4 import BeautifulSoup
import urllib.parse 
#from urllib.request import urlretrieve

datasource_url = 'https://jmlr.csail.mit.edu'
papers = []


def get_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html5lib')

    tags = soup.find_all('a')

    for tag in tags:
        if tag.has_attr('href') and 'html' in tag['href'] and 'papers' in tag['href']:
            abs_link = url + tag['href']

            abs_response = requests.get(abs_link)
            abs_soup = BeautifulSoup(abs_response.content, 'html5lib')
   
            keys = ['title', 'abstract', 'keywords', 'author', 'ref', 'datasource', 'datasource_url']

            keywords = []
            ref = abs_soup.find("meta", {"name":"citation_pdf_url"}).attrs["content"] 
            datasource = abs_soup.find("meta", {"name":"citation_journal_title"}).attrs["content"]
            authors_ = abs_soup.findAll("meta", {"name":"citation_author"})
            authors = [author["content"] for author in authors_]
            title = abs_soup.find("meta", {"name":"citation_title"}).attrs["content"]
            abstract_latex = abs_soup.find('p', class_= 'abstract').getText()

            # TODO add response time constraints
            pdf_response = requests.get(ref)
            #pdf_filename = urllib.parse.unquote(pdf_response.url).split('/')[-1].replace(' ', '_') 

            print('curently processing url {}'.format(ref))

            with io.BytesIO(pdf_response.content) as f:
                pdf = pdftotext.PDF(f)
                #print(pdf[0])
                # TODO not all papers have keywords
                try:
                    abstract_ = re.search(r'(?<=Abstract).*(?=Keywords)', pdf[0], re.DOTALL)
                # TODO several prerocessing steps can be taken into consideration
                    abstract_pdf = abstract_.group(0).replace('\n', ' ')
                    keywords_ = re.search(r'(?<=Keywords:).*(?=Contents|\n\n)', pdf[0], re.DOTALL)
                except:
                    abstract_pdf = []
                try:
                    keywords_ = keywords_.group(0)
                    keywords_ = keywords_.split(',')
                    keywords = [keyword_.replace('\n', '').strip() for keyword_ in keywords_]
                except:
                    keywords = []
                
            # here we have two choices for the abstract: abstract_latex & abstract_pdf
            vals = [title, abstract_latex, keywords, authors, ref, datasource, datasource_url]
            
            papers.append(dict(zip(keys, vals)))
         


def create_data_file():
    with open('data.txt', 'w') as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)    

#get_page('https://jmlr.csail.mit.edu')
create_data_file()