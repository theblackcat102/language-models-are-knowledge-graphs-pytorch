from utils import compress_attention, create_mapping, BFS, build_graph, is_word
from multiprocessing import Pool
import spacy
import en_core_web_md
import torch
from transformers import AutoTokenizer, BertModel, GPT2Model
from constant import invalid_relations_set


def process_matrix(attentions, layer_idx = -1, head_num = 0, avg_head=False, trim=True, use_cuda=True):
    if avg_head:
        if use_cuda:
            attn =  torch.mean(attentions[0][layer_idx], 0).cpu()
        else:
            attn = torch.mean(attentions[0][layer_idx], 0)
        attention_matrix = attn.detach().numpy()
    else:
        attn = attentions[0][layer_idx][head_num]
        if use_cuda:
            attn = attn.cpu()
        attention_matrix = attn.detach().numpy()

    attention_matrix = attention_matrix[1:-1, 1:-1]

    return attention_matrix

def bfs(args):
    s, end, graph, max_size, black_list_relation = args
    return BFS(s, end, graph, max_size, black_list_relation)


def check_relations_validity(relations):
    for rel in relations:
        if rel.lower() in invalid_relations_set or rel.isnumeric():
            return False
    return True

def global_initializer(nlp_object):
    global spacy_nlp
    spacy_nlp = nlp_object

def filter_relation_sets(params):
    triplet, id2token = params

    triplet_idx = triplet[0]
    confidence = triplet[1]
    head, tail = triplet_idx[0], triplet_idx[-1]
    if head in id2token and tail in id2token:
        head = id2token[head]
        tail = id2token[tail]
        relations = [ spacy_nlp(id2token[idx])[0].lemma_  for idx in triplet_idx[1:-1] if idx in id2token ]
        if len(relations) > 0 and check_relations_validity(relations) and head.lower() not in invalid_relations_set and tail.lower() not in invalid_relations_set:
            return {'h': head, 't': tail, 'r': relations, 'c': confidence }
    return {}

def parse_sentence(sentence, tokenizer, encoder, nlp, use_cuda=True):
    '''Implement the match part of MAMA

    '''
    tokenizer_name = str(tokenizer.__str__)

    inputs, tokenid2word_mapping, token2id, noun_chunks  = create_mapping(sentence, return_pt=True, nlp=nlp, tokenizer=tokenizer)

    with torch.no_grad():
        if use_cuda:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
        outputs = encoder(**inputs, output_attentions=True)
    trim = True
    if 'GPT2' in tokenizer_name:
        trim  = False

    '''
    Use average of last layer attention : page 6, section 3.1.2
    '''
    attention = process_matrix(outputs[2], avg_head=True, trim=trim, use_cuda=use_cuda)

    merged_attention = compress_attention(attention, tokenid2word_mapping)
    attn_graph = build_graph(merged_attention)

    tail_head_pairs = []
    for head in noun_chunks:
        for tail in noun_chunks:
            if head != tail:
                tail_head_pairs.append((token2id[head], token2id[tail]))

    black_list_relation = set([ token2id[n]  for n in noun_chunks ])

    all_relation_pairs = []
    id2token = { value: key for key, value in token2id.items()}

    with Pool(10) as pool:
        params = [  ( pair[0], pair[1], attn_graph, max(tokenid2word_mapping), black_list_relation, ) for pair in tail_head_pairs]
        for output in pool.imap_unordered(bfs, params):
            if len(output):
                all_relation_pairs += [ (o, id2token) for o in output ]

    triplet_text = []
    with Pool(10, global_initializer, (nlp,)) as pool:
        for triplet in pool.imap_unordered(filter_relation_sets, all_relation_pairs):
            if len(triplet) > 0:
                triplet_text.append(triplet)
    return triplet_text


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    nlp = en_core_web_md.load()
    selected_model = 'gpt2-medium'

    use_cuda = True


    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    encoder = GPT2Model.from_pretrained(selected_model)
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()

    target_file = [
        '../../Documents/KGERT-v2/datasets/squad_v1.1/train-v1.1.json',
        # '../../Documents/KGERT-v2/datasets/squad_v1.1/wiki_dev_2020-18.json',
        # '../../Documents/KGERT-v2/datasets/squad_v1/dev-v1.1.json',
    ]

    output_filename = [
        'train_v1.1.jsonl',
        # 'wiki_2020-18.jsonl',
        # 'dev-v1.1.jsonl',
    ]

    for target_file, output_filename in zip(target_file, output_filename):
        with open(target_file, 'r') as f:
            dataset = json.load(f)

        output_filename = selected_model +'_'+ output_filename

        print(target_file, output_filename)

        f = open(output_filename,'w')
        for data in tqdm(dataset['data'], dynamic_ncols=True):
            for para in data['paragraphs']:
                context = para['context']
                for sent in nlp(context).sents:
                    for output in parse_sentence(sent.text, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        f.write(json.dumps(output)+'\n')
                f.flush()

                for question in para['qas']:
                    question = question['question']
                    for output in parse_sentence(question, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        f.write(json.dumps(output)+'\n')
                f.flush()
        f.close()