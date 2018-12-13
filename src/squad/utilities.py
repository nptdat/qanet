from typing import List, Optional
import re
import spacy
import nltk
import numpy as np
import tensorflow as tf

def load_glove(filepath: str):
    """Load GLoVe data"""
    wv = {}
    with open(filepath, 'rt') as f:
        for line in f:
            tokens = line.strip().split(' ')
            wv[tokens[0]] = [float(t) for t in tokens[1:]]
    return wv


def prune_word_vectors(wv: dict, vocab: set):
    """
    Remove vectors of the words which does not appear in vocab

    Args:
        wv: word vectors, mapping from a string to an array of floats
        vocab: set of words

    Returns:
        Word vectors pruned from wv
    """
    # Extend the vocab to set of all lowercase-words
    exp_vocab = vocab.union({w.lower() for w in vocab})
    # The final vocabulary is the intersection between vocab accumulated from SQuAD data and all GLoVe words
    for w in list(wv.keys()):
        if w not in exp_vocab and w.lower() not in exp_vocab:
            del wv[w]
    return wv


def to_chars(tokens: List[str], token_len: int, pad_char: str):
    """
    Convert each token from a list into a list of chars.
    Args:
        tokens: list of strings
        token_len: fixed num of chars for each word
    Returns:
        char_tokens: List[List[str]]: For each token in the input, convert to a list of chars (fixed len). Increase 1 more dim with respect to the input.
    """
    def convert(tok):
        chars = list(tok)
        if len(chars) < token_len:
            chars += [pad_char] * (token_len - len(chars))
        return chars[:token_len]

    char_list = list([convert(tok) for tok in tokens])
    return char_list


def augment_long_text(context: List[str], answers: List[dict]):
    """
    Improve a context text for better tokenization based on all answers of that context.
    Motivation:
        - The context may be wrongly tokenized,
        Ex1: the sequence "...Beyoncé married Jay Z. She publicly..." is tokenized as [..., 'Beyoncé', 'married', 'Jay', 'Z.', 'She', 'publicly',...]
        (in the paragraph including the question '56be95823aeaaa14008c910c')
        Ex2: the sequence "grossing $68 million—$60 million more than Cadillac Records" is tokenized as [..., 'grossing', '$', '68', 'million—$60', 'million', 'more', 'than', 'Cadillac', 'Records',...]
        (in the paragraph including the question '56bf99aca10cfb14005511ab')

        - In some cases, answers cannot be matched with those wrong tokens.
        In Ex1, the answer ['Jay', 'Z'] is misaligned with ['Jay', 'Z.']
        In Ex2, the answer ['60', 'million'] is misaligned with ['million—$60', 'million']

    Solution:
        - Use tokens from all related answers to fix the wrong tokens.

    Args:
        context: raw text of the paragraph context
        answers: List of answers related to the context. Each answer is a dict which contains 2 keys: 'text' and 'answer_start' as in json raw answer data.

    Returns:
        Augmented context: context text with some spaces inserted to guide the tokenization.
    """
    # print(answers)
    answers = sorted(answers, key=lambda a: a['answer_start'], reverse=True)

    # Remove duplicated answers
    i = 0
    while i < len(answers)-1:
        if answers[i]['answer_start'] == answers[i+1]['answer_start']:
            answers.remove(answers[i+1])
        else:
            i += 1

    # Insert SPACE into context to guide the tokenizer.
    for answer in answers:
        start = answer['answer_start']
        end = start + len(answer['text'])
        if end < len(context)-1 and not context[end+1].isalnum() and context[end+1] != ' ':
            context = context[:end] + ' ' + context[end:]
        if start > 0 and not context[start-1].isalnum() and context[start-1] != ' ':
            context = context[:start] + ' ' + context[start:]

    return context



nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
# sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')
# word_tokenizer = nltk.TreebankWordTokenizer()
def tokenize(text):
    ##### Tokenization with NLTK
    # tokens = []
    # sents = sent_tokenzier.tokenize(text)
    # for sent in sents:
    #     tokens.extend(word_tokenizer.tokenize(sent))
    # return tokens

    ##### Tokenization with spacy
    doc = nlp(text)
    return list([tok.text for tok in doc if tok.text and tok.text != ' '])


# PATT = re.compile('\[[a-z]+ [0-9]+\]')
PATT = re.compile('\[[a-z0-9 ]+\]')
def tokenize_long_text(text):
    splitters = PATT.findall(text)
    if len(splitters) > 0:
        toks = []
        segments = PATT.split(text)
        if len(splitters) < len(segments):
            splitters.append('')
        for seg1, seg2 in zip(segments, splitters):
            for txt in [seg1, seg2]:
                if txt:
                    toks.extend(tokenize(txt))
        return toks
    else:
        return tokenize(text)



def align(context: str, context_toks: List[str]):
    """
    Align tokens with their original text

    Args:
        context: original text
        context_toks: list of tokens of the original text

    Return:
        anchors: list of index in the original text for each tokens,
                 i.e., context[anchors[i]] is the start position of the context_toks[i] in context.
    """
    curr = 0
    anchors = []
    for tok in context_toks:
        try:
            idx = context.index(tok, curr)
            if idx >= 0:
                anchors.append(idx)
                curr = idx + len(tok)
        except ValueError:
            print('Cannot align tokens with original text whe tokens: {}, orig text: {}'.format(tok, context[curr:]))
            return None
    return anchors


def get_batch(X_data, y_data=None, batch_size=32, shuffle=False):
    bs = batch_size
    y_data = y_data or []
    N = len(X_data[0])
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    num_batches = (N-1) // bs + 1
    for b in range(num_batches):
        idx_from = b * bs
        idx_to = min((b+1)*bs, N)
        X_batch = [X[idx[idx_from:idx_to]] for X in X_data]
        y_batch = [y[idx[idx_from:idx_to]] if len(y) > 0 else []  for y in y_data]
        yield (X_batch, y_batch)


MINUS_INFINITY = -1e30
def mask_logits(logits, mask):
    return logits + MINUS_INFINITY * (1 - tf.cast(mask, tf.float32))
