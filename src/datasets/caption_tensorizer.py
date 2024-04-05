import os.path as op
import torch
import re
import html


class CaptionTensorizer(object):

    def __init__(self,
                 tokenizer,
                 max_img_seq_length=50,
                 max_seq_length=70,
                 max_seq_a_length=40,
                 is_train=True,
                 tagger=None):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            attn_mask_type: attention mask type, support seq2seq/bidirectional/cap_s2s/cap_bidir.
            mask_b: whether to mask text_b or not during training.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.tagger = tagger

    def prepro_raw_txt(self, text):
        # in case there are html special characters
        text = html.unescape(text)
        # FIXME: quick hack for text with emoji, may adopt twitter tokenizer later
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text
    

    def tokenize_text_inputs(self,
                             text_a,
                             text_b=None,
                             cls_token_segment_id=0,
                             pad_token_segment_id=0,
                             sequence_a_segment_id=0,
                             sequence_b_segment_id=1,
                             text_meta=None):
        text_a = self.prepro_raw_txt(text_a)
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        #tokens = [self.tokenizer.cls_token
        #          ] + tokens_a + [self.tokenizer.sep_token]
        tokens = tokens_a
        segment_ids = [cls_token_segment_id
                       ] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            text_b = self.prepro_raw_txt(text_b)
            # pad text_a to keep it in fixed length for better inference.
            # we do not use pos tag for text_b
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[:(self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        return tokens, segment_ids, seq_a_len, seq_len


    def tensorize_example_e2e(self,
                              text_a,
                              img_feat,
                              text_b=None,
                              cls_token_segment_id=0,
                              pad_token_segment_id=0,
                              sequence_a_segment_id=0,
                              sequence_b_segment_id=1,
                              text_meta=None,
                              mode='default'):
        # tokenize the texts
        tokens, segment_ids, seq_a_len, seq_len = self.tokenize_text_inputs(
            text_a, text_b, cls_token_segment_id, pad_token_segment_id,
            sequence_a_segment_id, sequence_b_segment_id, text_meta)

        # pad on the right for image captioning
        seq_padding_len = self.max_seq_len - seq_len
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.padding_side = 'right'

        raw_tokens = tokens
        tokens = raw_tokens + ([self.tokenizer.pad_token] *
                                         seq_padding_len)
        input_tokens = raw_tokens[:-1] + (
            [self.tokenizer.pad_token] * (seq_padding_len + 1))
        output_tokens = raw_tokens[1:] + ([self.tokenizer.pad_token] *
                                          (seq_padding_len + 1))
        input_token_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(input_tokens),
            dtype=torch.long)
        output_token_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(output_tokens),
            dtype=torch.long)

        segment_ids += ([pad_token_segment_id] * seq_padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            #mlm_targets = torch.tensor(mlm_targets, dtype=torch.long)
            return (input_ids, segment_ids, img_feat,
                    input_token_ids, output_token_ids)
        return input_ids, segment_ids, img_feat, input_token_ids, output_token_ids

    

def build_tensorizer(args, tokenizer, is_train=True):
    if is_train:
        return CaptionTensorizer(
            tokenizer,
            max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length,
            max_seq_a_length=args.max_seq_a_length,
            is_train=True,
            tagger=None,
        )
    return CaptionTensorizer(
        tokenizer,
        max_img_seq_length=args.max_img_seq_length,
        max_seq_length=args.max_seq_length
        if args.add_od_labels else args.max_gen_length,
        max_seq_a_length=args.max_gen_length,
        is_train=False
    )