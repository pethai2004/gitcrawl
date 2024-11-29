# TODO : implement TQDM for progress bar and add more error handling and logging
import os 
import json
import pickle
import time
from io import BytesIO
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional, Generator
import requests 
import nbformat # type: ignore
from nbconvert import PythonExporter # type: ignore

headers = {"Authorization" : ""} # set token here or use set_header function
base_git = "https://api.github.com"

def set_header(token: str):
    global headers
    headers = {"Authorization" : token}
    
def jup_to_py(jup_content: str) -> str:
    '''Convert jupyter content to string of python content, remove the output'''
    # Parse the notebook
    notebook = nbformat.reads(jup_content, as_version=4)

    # Clear the outputs
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []

    # Convert to Python
    python_exporter = PythonExporter()
    py_content, _ = python_exporter.from_notebook_node(notebook)
    
    return py_content

class ScrapGIT:

    def __init__(self, 
                name_repos: Union[List[str], tuple[str]]=None, 
                save_dir: str="data_code",
                checkpoint_path: str="scraper_state.pickle",
                n_workers: int=32, 
                extension: tuple=("py", "ipynb")):
        '''
        name_repos: list of repository names to scrape
        save_dir: directory to save the scraped files
        checkpoint_path: path to save the checkpoint
        n_workers: number of workers to use
        extension: extension of the files to scrape
        
        '''
        self.name_repos = sorted(name_repos) if name_repos is not None else []
        self.n_workers = n_workers
        self.extension = extension
        self.save_dir = save_dir
        self.checkpoint_path = checkpoint_path
        self.buffer_size = 100 
        # buffer should not be too large since it will ignore max_file_size and exceed it, otherwise make max_file_size larger
        self.max_file_size = 5 # in gb
        self.state_idx, self.file_idx = 0, 0
        
    @property
    def size_retrieved(self):
        return os.path.getsize(self.save_dir)
    @property
    def name_repos_left(self):
        return self.name_repos[self.state_idx:]
    
    @staticmethod
    def load_scraper_state(pickle_path: str):
        # initialize the class from the saved state
        variables_state = pickle.load(open(pickle_path, "rb")) # dictionary of variables
        scraper = ScrapGIT()
        # set attributes
        for key, value in variables_state.items():
            setattr(scraper, key, value)
        return scraper
        
    def save_scraper_state(self, pickle_path: str):
        if len(self.name_repos) == 0:
            raise ValueError("No repositories to save")
        variables = {
            "name_repos" : self.name_repos,
            "extension" : self.extension,
            "save_dir" : self.save_dir,
            "n_workers" : self.n_workers,
            "max_file_size" : self.max_file_size,
            "state_idx" : self.state_idx,
            "file_idx" : self.file_idx
        } 
        pickle.dump(variables, open(pickle_path, "wb"))

    @staticmethod
    def _retrieve_file(repo_name: str, extension=("py", "ipynb"), apply_convert=True) -> Optional[dict]:
        repo_name = repo_name.strip("/")
        
        try:
            sess = requests.Session()
            link = f"{base_git}/repos/{repo_name}/zipball"
            response = request_get(link, session=sess) 
            if isinstance(response, str): 
                return None 

            byte_content = BytesIO(response.content)
            contents = []
            with zipfile.ZipFile(byte_content, "r") as zf:
                
                for finfo in zf.infolist():
                    if len(extension) > 0 and finfo.filename.endswith(tuple(extension)):
                            
                        with zf.open(finfo) as f:
                            content = f.read().decode("utf-8")
                            if len(content) == 0: continue
                            if apply_convert is True and finfo.filename.endswith("ipynb"):
                                content = jup_to_py(content)
                            contents.append(
                                {
                                    "repos": repo_name,
                                    "content": content,
                                    "length": len(content)
                                }
                            )
        except Exception as e: 
            return None 
        
        return contents
    
    def _retrieve_file_multi(self, inputs=None) -> Generator[dict, None, None]:
        ## should only be called once
        if inputs is None:
            inputs = self.name_repos
        
        with ThreadPoolExecutor(self.n_workers) as executor:
            futures = {executor.submit(self._retrieve_file, repo_name): repo_name for repo_name in inputs}
            
            for idx, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is None or len(result) == 0:
                    continue
                if idx % 500 == 0:
                    print(f"Processed success with {idx} / {len(inputs)}")
                for r in result:
                    yield r

    def save_raw_file(self, do_sleep: int=0):
        '''Save the raw files to the directory specified in save_dir'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        continued_inputs = self.name_repos_left
        if not hasattr(self, "_file_generator"):
            self._file_generator = self._retrieve_file_multi(continued_inputs)

        buffer = []
        
        for _, item in enumerate(self._file_generator):
            self.state_idx += 1# add the state index to the current index if continues
            if do_sleep > 0 and self.state_idx % 100 == 0: 
                time.sleep(do_sleep)   
                
            self.save_scraper_state(self.checkpoint_path)
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                current_path = os.path.join(self.save_dir, f"{str(self.file_idx).zfill(3)}.pickle")
                with open(current_path, "ab") as f:
                    for buf in buffer:
                        pickle.dump(buf, f) 

                buffer = []
                if os.path.getsize(current_path) > self.max_file_size * 1024 * 1024 * 1024: # should move this to buffer loop
                    print(f"File size exceeds {self.max_file_size} GB, creating new file")
                    self.file_idx += 1

            if self.state_idx % 100 == 0:
                size_dir = sum(os.path.getsize(os.path.join(self.save_dir, i)) for i in os.listdir(self.save_dir))
                print(f"Processed {self.state_idx} / {len(self.name_repos)} files, total size: {size_dir / 1024 / 1024 / 1024:.2f} GB, Time: {time.ctime()}")
                
        if len(buffer) > 0: # save the remaining buffer
            current_path = os.path.join(self.save_dir, f"{str(self.file_idx).zfill(3)}.pickle") # in case this is first iteration
            with open(current_path, "wb") as f:
                for buf in buffer:
                    pickle.dump(buf, f)
        
################################## UTILITIES ############################################## 

def request_get(link: str, 
                session: requests.Session=None, 
                max_retries: int=3, 
                backoff_factor: int=1,
                timeout: int=5,
                *args, **kwargs) -> Optional[requests.Response]:
    
    global headers
    '''Customized requests.get function'''
    session = session if session is not None else requests.Session()
    retry_strategy = requests.packages.urllib3.util.retry.Retry(
        total=max_retries,
        backoff_factor=backoff_factor,  
        status_forcelist=[403, 500, 502, 503, 504], # exclude 429
        raise_on_status=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    try: 
        response = session.get(link, timeout=timeout, headers=headers, *args, **kwargs)
        response.raise_for_status()
        return response
    
    except requests.exceptions.RequestException as e:
        return e
                
def file_load_generator(dir_path: str, max_sample=None, verbose=True) -> Generator[dict, None, None]:
    ''' This function is a generator that loads and yields data from pickle files in a given directory.
    Args:
        dir_path (str): The path to the directory containing the pickle files. If a path to a single pickle file is given, the function will load data from this file.
    Yields:
        return_data: The data loaded from the current pickle file.
    '''
    if not os.path.isdir(dir_path) and os.path.exists(dir_path):
        if dir_path.endswith(".pickle"):
            print(f"{dir_path} is not a directory but a file, loading file")
            paths = [dir_path]
        else:
            raise ValueError(f"{dir_path} is not a directory or a file")
    else:
        paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) \
        if f.endswith(".pickle")])
    
    idx_count = 0 
    for file in paths:
        if verbose:
            print(f"Recieving file: {file} with index: {idx_count} / {len(paths)}")
        with open(file, "rb") as f:
            while True:
                try:
                    return_data = pickle.load(f)
                    idx_count += 1
                    if max_sample is not None and idx_count >= max_sample:
                        break
                    yield return_data
                except EOFError: # this is the end of the file (always happens at the end of the file)
                    if verbose:
                        print(f"EOFError at file: {file}, for idx: {idx_count}")
                    break   

def save_json(data_input, dir_name, max_sample_per_file=100): # max_bytes is in GB
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    shard = []
    count = 0
    f = open(os.path.join(dir_name, f"{str(str(count)).zfill(3)}.json"), 'w')
    for data in data_input:
        shard.append(data)
        if len(shard) >= max_sample_per_file: #TODO: should be in according to byte instead of number of samples 
            json.dump({"data": shard}, f)
            f.close()
            shard = []
            count += 1
            f = open(os.path.join(dir_name, f"{str(str(count)).zfill(3)}.json"), 'w')
    json.dump({"data": shard}, f) # save the remaining data
    f.close()

