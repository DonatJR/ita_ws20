# from habanero import Crossref

# cr = Crossref()
# x = cr.members(ids = 320, works = True)

# for item in x["message"]["items"]:
#   doi = item["DOI"]
#   r = cr.works(ids=doi)
#   print(r["message"].keys())
  
# from crossref.restful import Works

# works = Works()
# for i in works.filter(member=320):
#   print(i.keys())

import json
import crossref_commons.retrieval
from crossref_commons.iteration import iterate_publications_as_json
from crossref_commons.sampling import get_sample

def get_sample():
    # TODO: check why this takes forever
    filter = {'has-abstract': 'true', 'type': 'journal-article'}
    queries = {'query': 'machine learning'}
    sample = get_sample(size=5, filter=filter, queries=queries)
    print(sample)
    

def get_data(count=5): 
    filter = {'has-abstract': 'true', 'type': 'journal-article'}
    queries = {} #{'query': 'machine learning'}
    
    try:
        publications = iterate_publications_as_json(max_results=count, filter=filter, queries=queries)
    except e:
        print('There was an error accessing the Crossref API')
    else:
        data = []
        datasource = "Crossref API",
        datasource_url = "https://api.crossref.org/"
        
        for p in publications:
            # if p['language'] != 'en':
            #     continue
            
            abstract = p['abstract']
            authors = []
            for author in p['author']:
                authors += [author['given'] + ' ' + author['family']]
                
            title = ''
            for t in p['title']:
                title = t
                break
            
            links = []
            for link in p['link']:
                links += [link]
            
            # TODO: find most relevant link in list
            ref = '' if len(links) == 0 else links[0]
            
            # TODO: extract keywords from pdf (in link) if available
            keywords = []
            
            data += [{"title": title, "abstract": abstract, "keywords": [], "author": authors, "ref": ref, "datasource": datasource, "datasource_url": datasource_url}]
            
        return data
      
print(get_data(1))