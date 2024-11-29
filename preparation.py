# TODO: implement de-duplication and filtering of code snippets (big)

# This script is designed for processing Python language dataset only which is used in training task
import pickle 
import tokenize 
from io import StringIO
import ast, astunparse
import os 
from multiprocessing import Pool 


def flatten_list(inputs):
    ''' flatten any any nested list, list of list, list of list of list, etc.'''
    for i in inputs:
        if isinstance(i, (list, tuple)):
            yield from flatten_list(i)
        else:
            yield i

def remove_duplicated_cls_def(text: str):
    '''Remove duplicated class definition.'''
    tree = ast.parse(text)
    function_names = set()
    new_tree_body = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name not in function_names:
                function_names.add(node.name)
                new_tree_body.append(node)
        else:
            new_tree_body.append(node)

    tree.body = new_tree_body
    return astunparse.unparse(tree)

def extract_vars(text: str, cd_only=True, 
                 include_var=False, 
                 include_import=False, 
                 include_cond=False, 
                 modules=None):
    
    '''Extract variables from a given text.'''
    text = remove_duplicated_cls_def(text)
    tree = ast.parse(text)
    code_snippets = []
    
    if modules is not None:

        imported_modules = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules:
                        imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module in modules:
                    imported_modules.append(node.module)
        
        if len(imported_modules) == 0:
            return None
    
    for node in tree.body:
            
        if cd_only: 
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                code_snippets.append(astunparse.unparse(node))
                
        else:  
            if (include_import and isinstance(node, (ast.Import, ast.ImportFrom))):
                code_snippets.append(astunparse.unparse(node).rstrip("\n"))
                    
            elif (include_cond and isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.AsyncFor))):
                code_snippets.append(astunparse.unparse(node))
                
            elif (include_var and isinstance(node, ast.Assign)):
                code_snippets.append(astunparse.unparse(node))
            
    return code_snippets

def em(sample, **kwargs):
    
    if isinstance(sample, dict):
        sample = sample["content"]
        
    code_snippets = []
    for content in sample:
        try:
            extracted = extract_vars(content, **kwargs)
            if extracted is not None:
                code_snippets.append(extracted)
        except:
            continue
    code_snippets = list(flatten_list(code_snippets))
    return code_snippets

def get_comment_and_doc(text, min_length=50):
    '''Get all comments and docstrings from a given code snippet.'''
    comments = []
    docstrings = []

    tokens = tokenize.generate_tokens(StringIO(text).readline)
    for token_type, token_string, _, _, _ in tokens:
        token_length = len(token_string)
        if token_length >= min_length:
            if token_type == tokenize.COMMENT:
                comments.append(token_string)
            elif token_type == tokenize.STRING:
                docstrings.append(token_string)
    
    return comments, docstrings

def load_data(path_to_data):
    """ Load data from a pickle file. """
    data = []
    with open(path_to_data, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    return data

def extract_multiple(file_dir, save_dir="extracted_dir", save_prefix="extracted", **kwargs):
    if os.path.exists(save_dir):
        raise ValueError(f"{save_dir} already exists.")
    os.makedirs(save_dir)
    
    files_list = os.listdir(file_dir)
    
    for file in files_list:
        if file.endswith(".pickle"):
            print(f"Processing {file}, size {os.path.getsize(os.path.join(file_dir, file))}")
            raw_data = load_data(os.path.join(file_dir, file))
            
            with Pool() as p:
                extracted = p.map(em, raw_data)
                
            extracted = list(flatten_list(extracted))
            with open(os.path.join(save_dir, save_prefix + "_" + file), "wb") as f:
                pickle.dump(extracted, f)
                print(f"Saved {len(extracted)} code with size {os.path.getsize(f.name)}")

