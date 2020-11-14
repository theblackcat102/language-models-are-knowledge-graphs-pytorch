import sys, os
from process import parse_sentence
from transformers import AutoTokenizer, BertModel, GPT2Model
import argparse
import en_core_web_md
from tqdm import tqdm
import json

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Process lines of text corpus into knowledgraph')
parser.add_argument('input_filename', type=str, help='text file as input')
parser.add_argument('output_filename', type=str, help='output text file')
parser.add_argument('--language_model',default='bert-base-cased', 
                    choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                    help='which language model to use')
parser.add_argument('--use_cuda', default=True, 
                        type=str2bool, nargs='?',
                        help="Use cuda?")
parser.add_argument('--include_text_output', default=False, 
                        type=str2bool, nargs='?',
                        help="Include original sentence in output")

args = parser.parse_args()

use_cuda = args.use_cuda
nlp = en_core_web_md.load()

'''Create
Tested language model:

1. bert-base-cased

2. gpt2-medium

Basically any model that belongs to this family should work

'''

language_model = args.language_model


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    if 'gpt2' in language_model:
        encoder = GPT2Model.from_pretrained(language_model)
    else:
        encoder = BertModel.from_pretrained(language_model)
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()    
    input_filename = args.input_filename
    output_filename = args.output_filename
    include_sentence = args.include_text_output

    with open(input_filename, 'r') as f, open(output_filename, 'w') as g:
        for idx, line in enumerate(tqdm(f)):
            sentence  = line.strip()
            if len(sentence):
                valid_triplets = []
                for sent in nlp(sentence).sents:
                    for triplets in parse_sentence(sent.text, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        valid_triplets.append(triplets)

                if len(valid_triplets) > 0:
                    output = { 'line': idx, 'tri': valid_triplets }
                    if include_sentence:
                        output['sent'] = sentence
                    g.write(json.dumps( output )+'\n')
