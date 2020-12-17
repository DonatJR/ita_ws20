"""
This is a scraper of information from research papers hosted on https://jmlr.csail.mit.edu.
The papers are ordered in volumes. Each volume has its own URL where all the papers from the volume are listed.

There are currently 21 (1 - 21) volumes available.

"""

import io
import re
import json
import requests
import pdftotext
from bs4 import BeautifulSoup

datasource_url = 'https://jmlr.csail.mit.edu'
papers = []

def get_papers(url):
    """
    This function is responsible scraping all the reasarch papers in a given volume.
    A dictionary with the following properties:
    
        title [str]: title of the reasearch paper
        abstract [str]: abstract of the research paper
        keywords [list: str]: listed keywords
        author [list: str]: authors of the research papers
        ref [str]: URL of the pdf of the paper
        datasource [str]: Journal of Machine Learning
        datasource_url [str]: https://jmlr.csail.mit.edu
    
    is created out of each research paper. The dictionary is then added to a list (papers) containing all the scraped papers from the given volumes.
    
    args:
    ---
    url [str]: the url of the volume to be scraped
    
    """


    print("currently fetching URL {}\n".format(url))

    try:
      response = requests.get(url, timeout=0.6)
    except requests.ConnectionError as e:
        print("{} could not be fetched due to a Connection Error. Make sure you are connected to Internet. Technical Details given below.\n".format(url))
        print(str(e))
        return
    except requests.Timeout as e:
        print("{} could not be fetched du to a Timeout Error\n".format(url))
        print(str(e))
        return

    soup = BeautifulSoup(response.content, 'html5lib')

    tags = soup.find_all('a')

    for tag in tags:
        if tag.has_attr('href') and 'html' in tag['href'] and 'papers' in tag['href']:
            abs_link = datasource_url + tag['href']

            abs_response = requests.get(abs_link)
            abs_soup = BeautifulSoup(abs_response.content, 'html5lib')
   
            keys = ['title', 'abstract', 'keywords', 'author', 'ref', 'datasource', 'datasource_url']

            keywords = []
            ref = abs_soup.find("meta", {"name":"citation_pdf_url"}).attrs["content"]
            datasource = abs_soup.find("meta", {"name":"citation_journal_title"}).attrs["content"]
            authors_ = abs_soup.findAll("meta", {"name":"citation_author"})
            authors = [author["content"] for author in authors_]
            title = abs_soup.find("meta", {"name":"citation_title"}).attrs["content"]
            abstract_latex = abs_soup.find('p', class_= 'abstract').getText().strip()
            # remove inline math latex notation, i.e. $...$
            abstract_latex = re.sub(r'(\${1,2})(?:(?!\1)[\s\S])*\1', '', abstract_latex)

            # name of the pdf with the paper
            pdf_filename = ref.split('/')[-1].replace(' ', '_')

            print("currently fetching {} \n".format(ref))

            try:
                pdf_response = requests.get(ref, timeout=0.6)
            except requests.ConnectionError as e:
                print("{} at {} could not be fetched due to a Connection Error. Make sure you are connected to Internet. Technical Details given below.\n".format(pdf_filename, ref))
                print(str(e))
                continue
            except requests.Timeout as e:
                print("{} at {} could not be fetched du to a Timeout Error\n".format(pdf_filename, ref))
                print(str(e))
                continue
       
            print('curently processing {}\n'.format(pdf_filename))

            with io.BytesIO(pdf_response.content) as f:
                pdf = pdftotext.PDF(f)

                try:
                # TODO regex not working for papers with no keywords
                    abstract_ = re.search(r'(?<=Abstract).*(?=Keywords)', pdf[0], re.DOTALL)
                # TODO several prerocessing steps can be taken into consideration
                    abstract_pdf = abstract_.group(0).replace('\n', ' ')
                    keywords_ = re.search(r'(?<=Keywords:)([\s\w\d-]*,)*', pdf[0], re.DOTALL)
                except:
                    print("Paper at {} seems to have no abstract\n".format(ref))
                    abstract_pdf = []
                try:
                    keywords_ = keywords_.group(0)
                    keywords_ = keywords_.split(',')
                    keywords = [keyword_.replace('\n', '').strip() for keyword_ in keywords_]
                    keywords = list(filter(None, keywords))
                except:
                    print("Paper at {} seems to have no keywords\n".format(ref))
                    keywords = []
                
            # here we have two choices for the abstract: abstract_latex & abstract_pdf
            vals = [title, abstract_latex, keywords, authors, ref, datasource, datasource_url]
            
            papers.append(dict(zip(keys, vals)))
        

def create_data_file(start_vol_nr, end_vol_nr):
    """
    This function is responsible for creating the .json file with the scraped data.
    The parameters are used in file name, i.e. data_jmlr_vol13-18.json
    """
    with open('data_jmlr_vol{}.json'.format(start_vol_nr) if start_vol_nr == end_vol_nr else 'data_jmlr_vol{}-{}.json'.format(start_vol_nr, end_vol_nr), 'w') as f:
        json.dump({"papers":papers}, f, ensure_ascii=False, indent=4)
    
def scrape_vol(vol_nr):
    get_papers('https://jmlr.csail.mit.edu/papers/v{}'.format(vol_nr))
  
def scrape_vols(start_vol_nr, end_vol_nr=None):
    """
    Call this function in oder to start scraping.
    
    args:
    ---
    start_vol_nr [int]: first volume to be scraped
    end_vol_nr [int]: last volume (inclusive) to be scraped. arg is optional if only one volume is to be scraped
    
    returns:
    ---
    data_jmlr_vol [json]: file with the extracted information
    
    """
    if end_vol_nr is None:
        end_vol_nr = start_vol_nr

    for vol_nr in range(start_vol_nr, end_vol_nr + 1):
        scrape_vol(vol_nr)
    create_data_file(start_vol_nr, end_vol_nr)
    


