import unittest
import json
from common import Vocabulary, Span, Answer, Question, Paragraph, SquadData
from utilities import tokenize, align

SQUAD_JSON_FILE = 'test/testdata/train-v2.0_tiny.json'

class TestVocabulary(unittest.TestCase):
    def test_vectorize(self):
        wv = {
            'the': [0, 0],
            'I': [0, 0],
            'You': [0, 0],
            'to': [0, 0],
            'go': [0, 0],
            'school': [0, 0]
        }
        char_vocab = {'t', 'h', 'e', 'I', 'Y', 'o', 'u', 'g', 's', 'l'}
        vocab = Vocabulary(wv, char_vocab)
        tokens = ['I', 'go', 'to', 'school']

        # Convert short context
        vect = vocab.vectorize(tokens, 10)
        self.assertEqual(len(vect), 10)
        for k in range(len(tokens)):
            self.assertEqual(vect[k], vocab.w2id[tokens[k]])
        self.assertEqual(vect[len(tokens):], [0]*(10-len(tokens)))

        # Prune long context
        vect = vocab.vectorize(tokens, 3)
        self.assertEqual(len(vect), 3)
        for k in range(3):
            self.assertEqual(vect[k], vocab.w2id[tokens[k]])


class TestSpan(unittest.TestCase):
    def test_allocate(self):
        # Small data
        context = "Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child."
        context_toks = tokenize(context)
        anchors = align(context, context_toks)
        query = ['Houston', ',', 'Texas']
        start_char = 19
        end_char = 32
        span = Span.allocate(anchors, start_char, end_char)
        self.assertEqual(span.start, 4)
        self.assertEqual(span.end, 6)
        for k in range(span.start, span.end+1):
            self.assertEqual(context_toks[k], query[k-span.start])

        # Real data
        context = "Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\"."
        context_toks = tokenize(context)
        anchors = align(context, context_toks)
        query = ['Dangerously', 'in', 'Love']
        start_char = 505
        end_char = 523
        span = Span.allocate(anchors, start_char, end_char)
        self.assertEqual(span.start, 108)
        self.assertEqual(span.end, 110)
        for k in range(span.start, span.end+1):
            self.assertEqual(context_toks[k], query[k-span.start])


class TestAnswer(unittest.TestCase):
    def test_parse_json(self):
        train_json = json.load(open(SQUAD_JSON_FILE, 'rt'))
        para_js = train_json['data'][0]['paragraphs'][0]
        context = para_js['context']
        context_toks = tokenize(context)
        anchors = align(context, context_toks)
        answer_json = para_js['qas'][0]['answers']
        answers = Answer.parse_json(answer_json, context, context_toks, anchors)
        self.assertEqual(answers[0].span.start, 56)
        self.assertEqual(answers[0].span.end, 59)
        self.assertEqual(answers[0].answer_toks, ['in', 'the', 'late', '1990s'])

        para_js = train_json['data'][0]['paragraphs'][3]
        context = para_js['context']
        context_toks = tokenize(context)
        anchors = align(context, context_toks)
        answer_json = para_js['qas'][8]['answers']
        answers = Answer.parse_json(answer_json, context, context_toks, anchors)
        # self.assertEqual(answers[0].span)

        self.assertIsNotNone(answers[0].span)
        print(answers[0].span)

class TestQuestion(unittest.TestCase):
    pass

class TestParagraph(unittest.TestCase):
    def test_parse_json(self):
        train_json = json.load(open(SQUAD_JSON_FILE, 'rt'))

        para_js = train_json['data'][0]['paragraphs'][0]
        para = Paragraph.parse_json(para_js, 0)
        self.assertEqual(sorted(list(para.local_word_vocab)), ['"', "'s", '(', ')', ',', '-', '.', '/biːˈjɒnseɪ/', '100', '1981', '1990s', '2003', '4', '?', 'American', 'Awards', 'Baby', 'Beyonce', 'Beyoncé', 'Billboard', 'Born', 'Boy', 'Carter', 'Child', 'Crazy', 'Dangerously', 'Destiny', 'Giselle', 'Grammy', 'Hot', 'Houston', 'How', 'In', 'Knowles', 'Love', 'Managed', 'Mathew', 'R&B', 'September', 'Texas', 'Their', 'What', 'When', 'Who', 'YON', 'a', 'actress', 'album', 'all', 'an', 'and', 'areas', 'artist', 'as', 'awards', 'became', 'become', 'becoming', 'bee', 'best', 'born', 'by', 'child', 'city', 'compete', 'competitions', 'dancing', 'debut', 'decade', 'did', 'earned', 'established', 'fame', 'famous', 'father', 'featured', 'first', 'five', 'for', 'girl', 'group', 'groups', 'grow', 'growing', 'have', 'her', 'hiatus', 'in', 'is', 'known', 'late', 'lead', 'leave', 'made', 'managed', 'many', 'name', 'number', 'of', 'one', 'performed', 'popular', 'producer', 'raised', 'record', 'release', 'released', 'rise', 'role', 'rose', 'saw', 'say', 'selling', 'she', 'singer', 'singing', 'singles', 'solo', 'songwriter', 'start', 'state', 'the', 'time', 'to', 'up', 'various', 'was', 'what', 'when', 'which', 'win', 'world', 'worldwide'])
        self.assertEqual(sorted(list(para.local_char_vocab)), ['', '"', '&', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '8', '9', '?', 'A', 'B', 'C', 'D', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'R', 'S', 'T', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'é', 'ɒ', 'ɪ', 'ˈ', 'ː'])
