
import transformers as tfm
import tensorflow as tf 
import datasets

import os 
from io import StringIO
import tokenize
import tokenizers

datasets.logging.set_verbosity(datasets.logging.ERROR)

def remove_non_code(source): # https://stackoverflow.com/questions/1769332/script-to-remove-python-comments-docstrings

    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]

        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
        # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:

                    if start_col > 0:

                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out

def get_simple_dataset(tokenizer=None, 
                       min_length=256, 
                       shuffle=None, 
                       return_tensors="tf", 
                       with_format='torch'):
    '''Get basic configured dataset for training. (minimal preprocessing)'''
    
    dataset_hgf = datasets.load_dataset("Owaner/CodeComputeX", streaming=True)
    
    if tokenizer == "default":
        tokenizer = tfm.AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
        tokenizer.pad_token = tokenizer.eos_token

    dataset_hgf = dataset_hgf.map(lambda x: {"code": remove_non_code(x["code"].strip())})
    dataset_hgf = dataset_hgf.filter(lambda x: tf.strings.length(x["code"]) >= min_length)
    
    if isinstance(tokenizer, (tfm.AutoTokenizer, tfm.PreTrainedTokenizerFast)): 
        dataset_hgf = dataset_hgf.map(lambda x: tokenizer(x["code"], 
                                                        return_attention_mask=True, 
                                                        return_tensors=return_tensors, 
                                                        padding=True, 
                                                        truncation=True),
                                    )
    if shuffle:
        dataset_hgf = dataset_hgf.shuffle(shuffle, buffer_size=10000)
        
    dataset_hgf = dataset_hgf.with_format(with_format)

    return dataset_hgf
        
def get_trained_tokenizer(dataset_iter: iter, 
                          push_to: str=None,
                          vocab_size=12_000, 
                          special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                          save_path="tokenizer"):
    
    if os.path.exists(save_path):
        print("Loading tokenizer from file.")
        tokenizer = tokenizers.Tokenizer.from_file(save_path)
    else:  
        tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]")) 
    
    tokenizer.normalizer = tokenizers.normalizers.Sequence([
        tokenizers.normalizers.NFKC(),
        tokenizers.normalizers.Lowercase()
    ])
    
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        tokenizers.pre_tokenizers.Punctuation(),
        tokenizers.pre_tokenizers.WhitespaceSplit()
    ])
    
    trainer = tokenizers.trainers.WordPieceTrainer(vocab_size=vocab_size, 
                                                   special_tokens=special_tokens)
    
    tokenizer.train_from_iterator(dataset_iter, trainer=trainer)
    
    tokenizer.save(save_path)
    
    if push_to:
        wrapped_tokenizer = tfm.PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        wrapped_tokenizer.push_to_hub(push_to, use_temp_dir=True)
        print(f"Pushed tokenizer to {push_to}.")
        
    return tokenizer

class dataset_provider:
    
    def __init__(self, 
                 max_sample=1_000_000, 
                 max_token=100_000_000,
                 batch_size=16,
                ):
        
        self.dataset: datasets.IterableDatasetDict = get_simple_dataset()
        
        if "test" not in self.dataset and "train" in self.dataset:
            self.dataset = self.dataset["train"].train_test_split(test_size=0.2)
            
        self.iter_dataset = self.dataset["train"].iter(batch_size=batch_size)
        self.max_sample = max_sample
        self.max_token = max_token
        self.batch_size = batch_size
    
    def __iter__(self): return self
    
    def __next__(self):
        if self.idx >= self.max_sample:
            raise StopIteration
        else:
            self.idx += self.batch_size
            return next(self.iter_dataset)
        
    @staticmethod
    def _get(self):
        '''Return this new dataset_provider object as iterable.'''
        return self.__init__(self.max_sample, self.max_token, self.batch_size)
    