
import os
from preparation import *

if __name__ == "__main__":
    
    data_dir = "data_code"
    size_before = os.path.getsize(data_dir)
    print(f"Size before extraction: {size_before/ 1e9 } gb")
    ###
    extract_multiple(file_dir=data_dir, save_dir="extracted_dir")
    ###
    print(f"Size after extraction: {os.path.getsize('extracted_dir') / 1e9} gb")