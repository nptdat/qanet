import os
from typing import List, Optional
import pickle
import numpy as np
from utilities import augment_long_text, tokenize, tokenize_long_text, to_chars, align
from config import Config as cf

PAD = 0 # TODO: choose appropriate index for these special chars
UNK = 1
DEFAULT = {'PAD': PAD, 'UNK': UNK}
DEFAULT_C = {'': PAD, 'UNK': UNK}

def word_lookup(w: str, table: dict, default=None):
    """
    Translate a word into a value by looking up from a dict.
    First priority is case-sensitive, then the next priority is case-insensitive.
    In case the word does not exist in the dict, a KeyError exception is raised or a default value is returned.

    Args:
        w: word to translate
        table: a dict to translate the word by looking up
        default: If not None, this is the value to return in case the word does not exist in the table.

    Returns:
        Translated value for the word by looking up the word into table.
    """
    if w in table:                  # Match word in case-sentitive mode is the first priority
        return table[w]
    elif w.lower() in table:        # Then, case-insensitive
        return table[w.lower()]
    else:
        if default is not None:
            return default
        else:
            raise KeyError('Key `{}` not found'.format(w))


def char_lookup(c: str, table: dict, default=None):
    """
    Translate a char into a value by looking up from a dict.

    Args:
        c: char to translate
        table: a dict to translate the char by looking up
        default: If not None, this is the value to return in case the char does not exist in the table.

    Returns:
        Translated value for the char by looking up the char into table.
    """
    if c in table:                  # Match word in case-sentitive mode is the first priority
        return table[c]
    else:
        if default is not None:
            return default
        else:
            raise KeyError('Key `{}` not found'.format(c))


class Vocabulary(object):
    def __init__(self, wv: dict, char_vocab: set):
        offset = len(DEFAULT)
        w2id = {w: idx+offset for idx, w in enumerate(wv.keys())}
        w2id.update(DEFAULT)
        id2w = {i:w for w, i in w2id.items()}

        c2id = {c: idx+offset for idx, c in enumerate(list(char_vocab))}
        c2id.update(DEFAULT_C)
        id2c = {i:c for c, i in c2id.items()}

        self.wv = wv
        self.emb_size = len(wv['the'])  # most common word that absolutely appears in the dict
        self.w2id = w2id    # mapping word to index
        self.id2w = id2w    # mapping index to word
        self.c2id = c2id    # mapping char to index
        self.id2c = id2c    # mapping index to char


    def vectorize(self, tokens: List[str], length: int):
        """
        Convert list of text tokens into list of indices
        """
        vect = [word_lookup(t, self.w2id, default=UNK) for t in tokens]
        vect = vect[:length]
        if len(vect) < length:
            vect.extend([PAD]*(length-len(vect)))
        return vect


    def vectorize_c(self, chars_list: List[List[str]], length: int, w_length: int):
        """
        Convert list of list of chars into list of index-based representation
        """
        vects = []
        PAD_VECT = [PAD]*w_length
        for chars in chars_list:
            vects.append([char_lookup(c, self.c2id, default=UNK) for c in chars])
        vects = vects[:length]
        while len(vects) < length:
            vects.append(PAD_VECT)
        return vects


    def get_embed_weights(self):
        """
        Build weights for a word embedding layer.
        Note that pre-trained word embedding is used, so no need to parameterize embed_size.

        Args:
            emb_size: Dim of the vectors
        Returns:
            [N, emb_size] matrix, where N is number of VOCAB + 1 (for pad)
        """
        emb_size = len(self.wv[list(self.wv.keys())[0]])
        weights = np.zeros((len(self.id2w), emb_size))
        for i, tok in self.id2w.items():
            if tok in self.wv:
                weights[i] = self.wv[tok]
            else:
                weights[i] = np.random.uniform(0.0, 1.0, [emb_size])
        return weights

    def get_char_embed_weights(self, emb_size=64):
        """
        Initialize weights for char embedding layer.

        Args:
            emb_size: Dim of the vectors
        Returns:
            [len(id2c), emb_size] matrix
        """
        weights = emb = np.random.uniform(0.0, 1.0, size=(len(self.id2c), emb_size))
        return weights

    @property
    def vocab_size(self):
        return len(self.w2id)

    def __getitem__(self, idx):
        """
        Get vector for a word.
        """
        if not isinstance(idx, str):
            raise ValueError('Index must be a string')
        return word_lookup(idx, self.wv, default=None)

    def __contains__(self, idx):
        if not isinstance(idx, str):
            raise ValueError('Index must be a string')
        return idx in self.wv or idx.lower() in self.wv


class Span(object):
    def __init__(self, start_idx: int, end_idx: int):
        self.start = start_idx                  # index of the start token in context
        self.end = end_idx                      # index of the end token in context

    @classmethod
    def allocate(cls, anchors: List[int], start_char: int, end_char: int):
        start_idx = 0
        while anchors[start_idx] < start_char:
            start_idx += 1
        if anchors[start_idx] > start_char:
            start_idx -= 1
        end_idx = start_idx
        while end_idx < len(anchors) and anchors[end_idx] <= end_char:
            end_idx += 1
        end_idx -= 1
        return Span(start_idx, end_idx)

    def __str__(self):
        return "({}, {})".format(self.start, self.end)


class Answer(object):
    def __init__(self, answer_text: str, answer_toks: List[str], span: Span, answer_start: int):
        self.answer_text = answer_text      # original answer text in JSON
        self.answer_toks = answer_toks      # tokens of the original answer text
        self.answer_chars = to_chars(answer_toks, cf.WORD_LEN, cf.PAD_CHAR)   # list of chars of the answer text
        self.span = span                    # The span (token-based index) of the answer in context
        self.answer_start = answer_start    # start character in original answer text

    def vectorize(self, vocab: Vocabulary):
        self.answer: List[int] = vocab.vectorize(self.answer_toks, cf.ANSWER_LEN)
        self.answer_c: List[List[int]] = vocab.vectorize_c(self.answer_chars, cf.ANSWER_LEN, cf.WORD_LEN)

    @classmethod
    def parse_json(cls, answers_js: List[dict], context: str, context_toks: List[str], anchors: List[int]):
        answers = []
        for ans in answers_js:
            ans_text = ans['text']
            ans_start = ans['answer_start']
            ans_toks = tokenize(ans_text)

            # Identify the span from context, ans_text & start index
            span = Span.allocate(anchors, ans_start, ans_start+len(ans_text)-1)
            answers.append(Answer(ans_text, ans_toks, span, ans_start))
        return answers


class Question(object):
    def __init__(self, question_text: str, ques_id: str, question: List[str], answers: List[Answer], plausible_answers: List[Answer]):
        self.question_text = question_text          # original question text in JSON
        self.question_toks = question               # tokens of the original question text
        self.question_chars = to_chars(question, cf.WORD_LEN, cf.PAD_CHAR)   # list of chars of the question text
        self.answers = answers                      # list of Answer object of the question
        self.ques_id = ques_id                      # id of the question in JSON
        self.plausible_answers = plausible_answers
        self.paragraph = None     # handle to the parent paragraph

    def set_paragraph(self, paragraph):
        self.paragraph = paragraph

    def vectorize(self, vocab: Vocabulary):
        self.question: List[int] = vocab.vectorize(self.question_toks, cf.QUERY_LEN)
        self.question_c: List[List[int]] = vocab.vectorize_c(self.question_chars, cf.QUERY_LEN, cf.WORD_LEN)
        for answer in self.answers:
            answer.vectorize(vocab)


class Paragraph(object):
    def __init__(self, raw_context: str, context_text: str, context_toks: List[str], questions: List[Question], para_idx: int, anchors: List[int]):
        self.raw_context = raw_context          # original context text in JSON
        self.context_text = context_text        # augmented from original context text with SPACES to guide the tokenization
        self.context_toks = context_toks        # tokens of the context text
        self.context_chars = to_chars(context_toks, cf.WORD_LEN, cf.PAD_CHAR)     # chars of the context
        self.questions = questions              # list of Question objects
        self.local_word_vocab = self._build_local_word_vocab()
        self.local_char_vocab = self._build_local_char_vocab()
        self.para_idx = para_idx   # Just for management & debug. Not used in experiment.
        self.anchors = anchors

    def _build_local_word_vocab(self):
        local_vocab = set()
        local_vocab = local_vocab.union(set(self.context_toks))
        for question in self.questions:
            local_vocab = local_vocab.union(set(question.question_toks))
            for answer in question.answers + question.plausible_answers:
                local_vocab = local_vocab.union(set(answer.answer_toks))
        return local_vocab


    def _build_local_char_vocab(self):
        def char_set(tokens):
            chars = set()
            for tok in tokens:
                chars = chars.union(set(tok))
            return chars

        char_vocab = set()
        char_vocab = char_vocab.union(char_set(self.context_chars))
        for question in self.questions:
            char_vocab = char_vocab.union(char_set(question.question_chars))
            for answer in question.answers + question.plausible_answers:
                char_vocab = char_vocab.union(char_set(answer.answer_chars))
        return char_vocab


    @classmethod
    def parse_json(cls, para_js: dict, para_idx: int):
        # Accumulate all answers' tokens first
        all_para_answers = []
        for q in para_js['qas']:
            if 'answers' in q:
                all_para_answers.extend([ans for ans in q['answers']])
            if 'plausible_answers' in q:
                all_para_answers.extend([ans for ans in q['plausible_answers']])

        # Improve the context for better tokenization
        raw_context = para_js['context']
        # context = augment_long_text(para_js['context'], all_para_answers)
        context = raw_context

        context_toks = tokenize_long_text(context)
        context_toks = [t.strip(' ') for t in context_toks]
        anchors = align(raw_context, context_toks)
        questions = []
        for q in para_js['qas']:
            question_text = q['question']
            q_toks = tokenize(question_text)
            ques_id = q['id']
            answers = Answer.parse_json(q['answers'], raw_context, context_toks, anchors) if 'answers' in q else []
            plausible_answers = Answer.parse_json(q['plausible_answers'], raw_context, context_toks, anchors) if 'plausible_answers' in q else []
            questions.append(Question(question_text, ques_id, q_toks, answers, plausible_answers))

        para = Paragraph(raw_context, context, context_toks, questions, para_idx, anchors)
        for ques in questions:
            ques.set_paragraph(para)
        return para


    def vectorize(self, vocab):
        """
        Vectorize pargraph context, question text & answer text based on given vocab.
        """
        self.context: List[int] = vocab.vectorize(self.context_toks, cf.CONTEXT_LEN)
        self.context_c: List[List[int]] = vocab.vectorize_c(self.context_chars, cf.CONTEXT_LEN, cf.WORD_LEN)
        for question in self.questions:
            question.vectorize(vocab)




def exact_match(gt_s, gt_e, pr_s, pr_e):
    """
    Evaluate exact match of a predicted span over a ground truth span.
    Args:
        gt_s: index of the ground truth start position
        gt_e: index of the ground truth end position
        pr_s: index of the predicted start position
        pr_e: index of the predicted end position
    """
    return gt_s == pr_s and gt_e == pr_e



def f1(gt_s, gt_e, pr_s, pr_e):
    """
    Evaluate F1 score of a predicted span over a ground truth span.
    Args:
        gt_s: index of the ground truth start position
        gt_e: index of the ground truth end position
        pr_s: index of the predicted start position
        pr_e: index of the predicted end position
    """
    gt = {idx for idx in range(gt_s, gt_e+1)}
    pr = {idx for idx in range(pr_s, pr_e+1)}
    intersection = gt.intersection(pr)
    prec = 1. * len(intersection) / len(pr)
    rec =  1. * len(intersection) / len(gt)
    f1_score = (2. * prec * rec) / (prec+rec) if prec+rec != 0. else 0.
    return f1_score


def get_score(metric, gt_starts, gt_ends, pred_start, pred_end):
    """
    Args:
        metric: a metric function to calculate the score (exact_match or f1_score)
        gt_starts: (list) an array of start indices of the available answers
        gt_ends: (list) an array of end indices of the available answers
        pred_start: (int) predicted start index returned by a model
        pred_end: (int) predicted end index returned by a model
    Returns:
        The best score of the metric evaluated on multiple answer spans.
    """
    scores = []
    for gt_s, gt_e in zip(gt_starts, gt_ends):
        scores.append(metric(gt_s, gt_e, pred_start, pred_end))
    return 1.0 * np.max(scores)


class SquadData(object):
    """
    To save the whole object to pickle file:
    ```python
    data.save('data/squad_processed.pkl')
    ```

    To load the whole object from pickle file, and extract train & validation data
    ```python
    data = SquadData.load('data/squad_processed.pkl')
    ques_ids_train, X_train, y_train = data.train_data()
    ques_ids_valid, X_valid, y_valid = data.validation_data()
    ```

    To save structured data to binary files for fast loading:
    ```python
    data.save(np_path='data/numpy')
    ```

    To load numpy data from binary files:
    ```python
    word_vectors, char_vectors, ques_ids_train, X_train, y_train, ques_ids_valid, X_valid, y_valid = SquadData.load(np_path='data/numpy')
    ```
    """

    def __init__(self, train_paragraphs: List[Paragraph], dev_paragraphs: List[Paragraph], vocab: Vocabulary, squad_words: set, squad_chars: set):
        """
        Initializer.

        Args:
            train_paragraphs: list of Paragraph objects from train data
            dev_paragraphs: list of Paragraph objects from dev data
            vocab: Vocabulary object which store vectors of words appearing in Squad data
            squad_words: set of all tokens appearing in Squad data (context, question text, answer text).
                         Note that some tokens may not appear in vocab. They are treated as unknown words.
                         Note that this is a set of words, so it must not be used to map words to indices. Use Vocabulary.w2id instead.
            squad_chars: set of all characters appearing in Squad data (context, question text, answer text).
        """
        self.train_paragraphs = train_paragraphs
        self.dev_paragraphs = dev_paragraphs
        self.vocab = vocab
        self.squad_words = squad_words
        self.squad_chars = squad_chars

    def summary(self):
        print('Num of train paragraphs: {}'.format(len(self.train_paragraphs)))
        print('Num of dev paragraphs: {}'.format(len(self.dev_paragraphs)))
        print('Num words in vocab: {}'.format(self.vocab.vocab_size))
        print('Num unique words: {}'.format(len(self.squad_words)))
        print('Num unique chars: {}'.format(len(self.squad_chars)))
        unknown_words = [w for w in self.squad_words if w not in self.vocab]
        print('Num of unknown words: {}'.format(len(unknown_words)))


    def _generate_data(self, paragraphs, dataset: str ='train'):
        ques_ids = []
        contextw_inp, queryw_inp, contextc_inp, queryc_inp = [], [], [], []
        p1, p2, start, end = [], [], [], []
        long_count = 0

        for para in paragraphs:
            for ques in para.questions:
                if dataset == 'train':
                    for ans in ques.answers:
                        if ans.span.start >= cf.CONTEXT_LEN or ans.span.end >= cf.CONTEXT_LEN:
                            # print('ques.ques_id:', ques.ques_id, ',', 'ans.span.start, end:', ans.span.start, ',', ans.span.end)
                            long_count += 1
                            continue

                        ques_ids.append(ques.ques_id)

                        contextw_inp.append(para.context)
                        queryw_inp.append(ques.question)
                        contextc_inp.append(para.context_c)
                        queryc_inp.append(ques.question_c)

                        vect = np.zeros(cf.CONTEXT_LEN, dtype=np.float16)
                        vect[ans.span.start] = 1.
                        p1.append(vect)

                        vect = np.zeros(cf.CONTEXT_LEN, dtype=np.float16)
                        vect[ans.span.end] = 1.
                        p2.append(vect)

                        start.append(ans.span.start)
                        end.append(ans.span.end)
                else:   # dev dataset
                    ques_ids.append(ques.ques_id)

                    contextw_inp.append(para.context)
                    queryw_inp.append(ques.question)
                    contextc_inp.append(para.context_c)
                    queryc_inp.append(ques.question_c)

                    start_list = []
                    end_list = []
                    for ans in ques.answers:
                        if ans.span.start >= cf.CONTEXT_LEN or ans.span.end >= cf.CONTEXT_LEN:
                            long_count += 1
                            continue
                        start_list.append(ans.span.start)
                        end_list.append(ans.span.end)

                    # p1, p2 are ignored in dev set
                    start.append(start_list)
                    end.append(end_list)


        print('There are {} long answers'.format(long_count))
        ques_ids = np.array(ques_ids)
        contextw_inp, queryw_inp, contextc_inp, queryc_inp = np.array(contextw_inp), np.array(queryw_inp), np.array(contextc_inp), np.array(queryc_inp)
        p1, p2, start, end = np.array(p1), np.array(p2), np.array(start), np.array(end)
        return (ques_ids, [contextw_inp, queryw_inp, contextc_inp, queryc_inp], [p1, p2, start, end])


    def train_data(self):
        return self._generate_data(self.train_paragraphs)


    def validation_data(self):
        return self._generate_data(self.dev_paragraphs, dataset='dev')


    def search_paragraph(self, para_idx: int, dataset: str ='train'):
        """
        Search for paragraph by index. This function is used for debug only.
        """
        paragraphs = self.train_paragraphs if dataset == 'train' else self.dev_paragraphs
        for para in paragraphs:
            if para.para_idx == para_idx:
                return para
        return None


    def search_question(self, ques_id: str, dataset: str ='train'):
        """
        Search for question by ques_id. This function is used for debug only.
        """
        paragraphs = self.train_paragraphs if dataset == 'train' else self.dev_paragraphs
        for para in paragraphs:
            for ques in para.questions:
                if ques.ques_id == ques_id:
                    return ques
        return None




    @classmethod
    def evaluate(cls, gt_start_list, gt_end_list, pred_starts, pred_ends):
        """
        Evaluate ExactMatch score  & F1 score of predictions on a validation set.
        Args:
            gt_start_list: list of start indices of multiple ground-truth answer spans
            gt_start_list: list of end indices of multiple ground-truth answer spans
            pred_starts: list of predicted start indices
            pred_ends: list of predicted end indices

        Returns:
            A hash with 2 keys: 'exact_match' & 'f1'
        """
        em_score = 0
        f1_score = 0
        total = 0
        for gt_starts, gt_ends, pred_start, pred_end in zip(gt_start_list, gt_end_list, pred_starts, pred_ends):
            if len(gt_starts) > 0:
                em_score += get_score(exact_match, gt_starts, gt_ends, pred_start, pred_end)
                f1_score += get_score(f1, gt_starts, gt_ends, pred_start, pred_end)
            # If gt_starts is empty, the ground-truth answer is over the limit length of the input text.
            # We give penalty for that case, that means we give 0 to EM & F1 while we increase the total.
            total += 1

        em_score = 100. * em_score / total
        f1_score = 100. * f1_score / total
        em_score, f1_score
        return {
            'exact_match': em_score,
            'f1': f1_score
        }


    def save(self, filepath=None, np_path=None):
        def save_data(prefix, ques_ids,
                      contextw, queryw, contextc, queryc,
                      p1, p2, start, end):
            np.save(np_path + '/%s_ques_ids.npy' % prefix, ques_ids)
            np.save(np_path + '/%s_contextw.npy' % prefix, contextw)
            np.save(np_path + '/%s_queryw.npy' % prefix, queryw)
            np.save(np_path + '/%s_contextc.npy' % prefix, contextc)
            np.save(np_path + '/%s_queryc.npy' % prefix, queryc)
            np.save(np_path + '/%s_p1.npy' % prefix, p1)
            np.save(np_path + '/%s_p2.npy' % prefix, p2)
            np.save(np_path + '/%s_start.npy' % prefix, start)
            np.save(np_path + '/%s_end.npy' % prefix, end)


        if filepath:    # Save the SquadData object to pickle file (slow)
            print('Saving squad data to {}...'.format(filepath))
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:           # Save the binary data to *.npy files (fast)
            print('Accumulating train & validation arrays from the structure...')
            t_ques_ids, X_train, y_train = self.train_data()
            v_ques_ids, X_valid, y_valid = self.validation_data()

            t_contextw, t_queryw, t_contextc, t_queryc = X_train
            t_p1, t_p2, t_start, t_end = y_train
            v_contextw, v_queryw, v_contextc, v_queryc = X_valid
            v_p1, v_p2, v_start, v_end = y_valid

            if not os.path.exists(np_path):
                os.makedirs(np_path)

            print('Saving word vectors into numpy files...')
            word_vectors = self.vocab.get_embed_weights()
            char_vectors = self.vocab.get_char_embed_weights()
            np.save(np_path + '/word_vectors.npy', word_vectors)
            np.save(np_path + '/char_vectors.npy', char_vectors)

            print('Saving train arrays into numpy files...')
            save_data(
                'train', t_ques_ids,
                t_contextw, t_queryw, t_contextc, t_queryc,
                t_p1, t_p2, t_start, t_end)

            print('Saving validation arrays into numpy files...')
            save_data(
                'val', v_ques_ids,
                v_contextw, v_queryw, v_contextc, v_queryc,
                v_p1, v_p2, v_start, v_end)


    @classmethod
    def load(cls, filepath=None, np_path=None):
        def load_data(prefix):
            ques_ids = np.load(np_path + '/%s_ques_ids.npy' % prefix)
            contextw = np.load(np_path + '/%s_contextw.npy' % prefix)
            queryw = np.load(np_path + '/%s_queryw.npy' % prefix)
            contextc = np.load(np_path + '/%s_contextc.npy' % prefix)
            queryc = np.load(np_path + '/%s_queryc.npy' % prefix)
            p1 = np.load(np_path + '/%s_p1.npy' % prefix)
            p2 = np.load(np_path + '/%s_p2.npy' % prefix)
            start = np.load(np_path + '/%s_start.npy' % prefix)
            end = np.load(np_path + '/%s_end.npy' % prefix)
            return ques_ids, contextw, queryw, contextc, queryc, p1, p2, start, end

        if filepath:    # Load SquadData object from pickle file (slow)
            print('Loading squad data from pickle file {}...'.format(filepath))
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:           # Load binary data from *.npy files (fast)
            print('Loading word vectors from numpy files...')
            word_vectors = np.load(np_path + '/word_vectors.npy')
            char_vectors = np.load(np_path + '/char_vectors.npy')

            print('Loading train arrays from numpy files...')
            t_ques_ids, t_contextw, t_queryw, t_contextc, t_queryc, t_p1, t_p2, t_start, t_end = load_data('train')

            print('Loading validation arrays from numpy files...')
            v_ques_ids, v_contextw, v_queryw, v_contextc, v_queryc, v_p1, v_p2, v_start, v_end = load_data('val')

            return [
                word_vectors,
                char_vectors,
                t_ques_ids,
                [t_contextw, t_queryw, t_contextc, t_queryc],
                [t_p1, t_p2, t_start, t_end],
                v_ques_ids,
                [v_contextw, v_queryw, v_contextc, v_queryc],
                [v_p1, v_p2, v_start, v_end]
            ]
