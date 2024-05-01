import torch

from src.transformers import GPT2Tokenizer
from src.utils.logger import LOGGER as logger

class TokenizerHandler:
    
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    

    def tokenize_caption(self, args, caption, img_ID):

        tokenized_caption = self.tokenizer.encode(caption[0], max_length=args.max_seq_length, pad_to_max_length=True, 
                                                  return_tensors='pt', add_prefix_space=True)
        
        #workaround for token shifting
        add = torch.tensor([-100])
        tokenized_caption = torch.cat((add, tokenized_caption[0]), dim=0)

        #padded_caption = self.caption_padding(args, tokenized_caption, img_ID)

        proc_tokens = tokenized_caption.to(args.device)

        return proc_tokens


    def caption_padding(self, args, tokenized_caption, img_ID):

        if len(tokenized_caption) > args.max_seq_length:
            logger.info(f'Caption {img_ID} is longer than max sequence length {args.max_seq_length}. Truncating the caption to {args.max_seq_length}.')
            tokenized_caption = tokenized_caption[:args.max_seq_length]
    
    
        # Pad the tokenized caption to max_seq_length
        pad_length = (args.max_seq_length + 1) - len(tokenized_caption)
        pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
        padding = torch.full((pad_length,), pad_token, dtype=torch.long)
        padded_tokens = torch.cat((tokenized_caption, padding), dim=0)

        return padded_tokens