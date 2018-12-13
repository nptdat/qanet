import time
from tqdm import tqdm
import nltk
import json
from common import Vocabulary, Span, Answer, Question, Paragraph, SquadData
from config import Config as cf
from utilities import load_glove, prune_word_vectors


def parse_squad(data):
    def paragraph_iter(json_data):
        for art in json_data:
            for para in art['paragraphs']:
                yield para

    total_para = sum([len(art['paragraphs']) for art in data])
    paragraphs = []
    for para_idx, para in enumerate(tqdm(paragraph_iter(data), total=total_para)):
        paragraph = Paragraph.parse_json(para, para_idx)
        paragraphs.append(paragraph)
    return paragraphs


if __name__ == '__main__':
    tic = time.time()

    print('Loading SQuAD data...')
    train_json = json.load(open(cf.TRAIN_JSON, 'rt'))
    dev_json = json.load(open(cf.DEV_JSON, 'rt'))

    print('Loading GloVE data...')
    wv = load_glove(cf.EMBEDDING_FILE)

    print('Parsing train data...')
    train_paragraphs = parse_squad(train_json['data'])
    print('Parsing dev data...')
    dev_paragraphs = parse_squad(dev_json['data'])

    # Accumulate global vocabulary
    print('Building vocabulary...')
    accu_word_vocab = set()
    accu_char_vocab = set()
    for idx, para in enumerate(tqdm(train_paragraphs + dev_paragraphs)):
        accu_word_vocab = accu_word_vocab.union(para.local_word_vocab)
        accu_char_vocab = accu_char_vocab.union(para.local_char_vocab)

    # Vectorize the text tokens
    accu_word_vocab
    wv = prune_word_vectors(wv, accu_word_vocab)
    vocab = Vocabulary(wv, accu_char_vocab)
    for idx, para in enumerate(tqdm(train_paragraphs + dev_paragraphs)):
        para.vectorize(vocab)

    # Write data to file
    data = SquadData(train_paragraphs, dev_paragraphs, vocab, accu_word_vocab, accu_char_vocab)
    # Save the whole SquadData to pickle file
    data.save(cf.SQUAD_DATA)

    # Export binary files for fast loading
    data.save(np_path=cf.SQUAD_NP_DATA)

    print('Elapsed time: {}'.format(time.time() - tic))
