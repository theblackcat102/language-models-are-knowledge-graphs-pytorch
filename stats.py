from utils import compress_attention, create_mapping, BFS, build_graph, is_word
from multiprocessing import Pool
import spacy
import en_core_web_sm
import torch
from transformers import AutoTokenizer, BertModel

nlp = en_core_web_sm.load()

if __name__ == '__main__':
    import json
    from tqdm import tqdm

    target_file = [
        '../../Documents/KGERT-v2/datasets/squad_v1.1/wiki_dev_2020-18.json',
        '../../Documents/KGERT-v2/datasets/squad_v1/dev-v1.1.json',
        '../../Documents/KGERT-v2/datasets/squad_v1.1/train-v1.1.json',
    ]

    with open('stats.txt', 'a') as g:
        for target_file in target_file:
            with open(target_file, 'r') as f:
                dataset = json.load(f)
            print(target_file)

            sentence_cnt = 0
            word_cnt = 0
            for data in tqdm(dataset['data'], dynamic_ncols=True):
                for para in data['paragraphs']:
                    context = para['context']
                    doc = nlp(context)
                    sentence_cnt +=  len(list(doc.sents))
                    word_cnt += len(list(doc))

                    for question in para['qas']:
                        question = question['question']
                        doc = nlp(question)
                        sentence_cnt +=  len(list(doc.sents))
                        word_cnt += len(list(doc))

            print('sentence : %d' % sentence_cnt)
            print('word     : %d' % word_cnt)
        
            g.write(target_file+'\n')
            g.write('sentence : %d\n' % sentence_cnt)
            g.write('word     : %d\n' % word_cnt)
