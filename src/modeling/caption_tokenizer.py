import torch

from src.transformers import GPT2Tokenizer
from src.utils.logger import LOGGER as logger

class TokenizerHandler:
    
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    

    def tokenize_caption(self, args, caption):

        tokenized_captions = self.tokenizer.batch_encode_plus(
            caption, 
            padding='max_length',         # Pads to max_length
            truncation=True,              # Truncates to max_length
            max_length=args.max_seq_length,
            return_tensors='pt',
            add_prefix_space=True,
            pad_to_max_length=True
    )
        
        #workaround for token shifting
        prepend_token = -100
        add = prepend_token * torch.ones((tokenized_captions['input_ids'].size(0), 1), dtype=torch.long)
        tokenized_captions = torch.cat([add, tokenized_captions['input_ids']], dim=1)

        proc_tokens = tokenized_captions.to(args.device)

        return proc_tokens
