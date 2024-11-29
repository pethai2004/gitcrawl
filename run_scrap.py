
import requests
import gzip
import io
import json
import time 
from scraper import *

def load_paper_with_code_meta(exclude_non_official=True):
    r = requests.get("https://production-media.paperswithcode.com/about/links-between-papers-and-code.json.gz")
    data = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read()
    data = json.loads(data)
    data = filter(lambda x: "github" in x["repo_url"], data)
    if exclude_non_official:
        data = filter(lambda x: x["is_official"], data)
        
    data_f = [{"repos_name" : i["repo_url"].strip("https://github.com/"), 
                "paper_url_pdf": i["paper_url_pdf"], 
               "repo_url": i["repo_url"], 
               "paper_title": i["paper_title"]} for i in data]
    # filter only unique repos_name key in list of dict
    seen = set()
    new_data_f = []
    for d in data_f:
        if d['repos_name'] not in seen and d['repos_name'] != "":
            seen.add(d['repos_name'])
            new_data_f.append(d)
    
    return new_data_f

def load_from_searcH(path="results.json"):
    
    return json.load(open(path, "r"))

def load_combined(path_search="results.json"):
    
    data_paper = load_paper_with_code_meta()
    data_search = load_from_searcH(path_search)['items']
    
    data_paper = {i["repos_name"]: i for i in data_paper}
    data_search = {i["name"]: i for i in data_search}
    
    data_name_combined = set(data_paper).union(set(data_search))
    return data_name_combined

def Save_default():
    token = "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"
    set_header(token)
    
    # data_name = load_combined()
    data_name = load_paper_with_code_meta()
    data_name = [i["repos_name"] for i in data_name]
    
    start_time = time.time()
    
    try:
        path_scraper = "scraper_state.pickle"
        scp = ScrapGIT.load_scraper_state(path_scraper)
        print(f"Loaded state from {path_scraper}, continue scraping ...")
        
    except FileNotFoundError:
        scp = ScrapGIT(name_repos=data_name, 
                        save_dir="data_code", 
                        checkpoint_path="scraper_state.pickle",
                        n_workers=32)
        scp.max_file_size = 3
        scp.buffer_size = 100
    
    print(f"Number of repositories left: {len(scp.name_repos_left)} / {len(data_name)}")

    scp.save_raw_file(do_sleep=5)
        
    done_time = time.time() - start_time
    print(f"Time taken: {done_time / 60} minutes")
    
    
if __name__ == "__main__":
    
    Save_default()