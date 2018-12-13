import unittest
from utilities import prune_word_vectors, to_chars, tokenize, augment_long_text, tokenize_long_text
from config import Config as cf

class TestUtilities(unittest.TestCase):
    def test_load_glove(self):
        pass

    def test_prune_word_vectors(self):
        vocab = set({'I', 'go', 'to', 'school'})
        wv = {
            'I': [0, 0],
            'You': [0, 0],
            'to': [0, 0]
        }
        pruned_wv = prune_word_vectors(wv, vocab)
        self.assertEqual(sorted(list(pruned_wv.keys())), ['I', 'to'])


    def test_to_chars(self):
        chars = to_chars(['I', 'go', 'to', 'schools'], 3, cf.PAD_CHAR)
        self.assertEqual(chars, [
            ['I', '', ''],
            ['g', 'o', ''],
            ['t', 'o', ''],
            ['s', 'c', 'h']
        ])


    def test_tokenize(self):
        self.assertEqual(tokenize("What is the title of his first commercially successful work?"), ['What', 'is', 'the', 'title', 'of', 'his', 'first', 'commercially', 'successful', 'work', '?'])
        self.assertEqual(tokenize("Rondo Op. 1."), ['Rondo', 'Op', '.', '1', '.'])

    def test_augment_long_text(self):
        context = 'Although the film received negative reviews from critics, the movie did well at the US box office, grossing $68 million—$60 million more than Cadillac Records—on a budget of $20 million. The fight scene finale between Sharon and the character played by Ali Larter also won the 2010 MTV Movie Award for Best Fight.'
        answers = [
            {
                'text': '60 million',
                'answer_start': 121
            },
            {
                'text': 'MTV Movie Award for Best Fight',
                'answer_start': 282
            }
        ]
        result = augment_long_text(context, answers)
        self.assertEqual(result, 'Although the film received negative reviews from critics, the movie did well at the US box office, grossing $68 million—$ 60 million more than Cadillac Records—on a budget of $20 million. The fight scene finale between Sharon and the character played by Ali Larter also won the 2010 MTV Movie Award for Best Fight.')


    def test_tokenize_long_text(self):
        self.assertEqual(tokenize_long_text('aaa [n 3] abc def [web 1] xyz[n 15]'), ['aaa', '[', 'n', '3', ']', 'abc', 'def', '[', 'web', '1', ']', 'xyz', '[', 'n', '15', ']'])
        self.assertEqual(tokenize_long_text("From September 1823 to 1826 Chopin attended the Warsaw Lyceum, where he received organ lessons from the Czech musician Wilhelm W\u00fcrfel during his first year. In the autumn of 1826 he began a three-year course under the Silesian composer J\u00f3zef Elsner at the Warsaw Conservatory, studying music theory, figured bass and composition.[n 3] Throughout this period he continued to compose and to give recitals in concerts and salons in Warsaw. He was engaged by the inventors of a mechanical organ, the \"eolomelodicon\", and on this instrument in May 1825 he performed his own improvisation and part of a concerto by Moscheles. The success of this concert led to an invitation to give a similar recital on the instrument before Tsar Alexander I, who was visiting Warsaw; the Tsar presented him with a diamond ring. At a subsequent eolomelodicon concert on 10 June 1825, Chopin performed his Rondo Op. 1. This was the first of his works to be commercially published and earned him his first mention in the foreign press, when the Leipzig Allgemeine Musikalische Zeitung praised his \"wealth of musical ideas\"."), ['From', 'September', '1823', 'to', '1826', 'Chopin', 'attended', 'the', 'Warsaw', 'Lyceum', ',', 'where', 'he', 'received', 'organ', 'lessons', 'from', 'the', 'Czech', 'musician', 'Wilhelm', 'Würfel', 'during', 'his', 'first', 'year', '.', 'In', 'the', 'autumn', 'of', '1826', 'he', 'began', 'a', 'three', '-', 'year', 'course', 'under', 'the', 'Silesian', 'composer', 'Józef', 'Elsner', 'at', 'the', 'Warsaw', 'Conservatory', ',', 'studying', 'music', 'theory', ',', 'figured', 'bass', 'and', 'composition', '.', '[', 'n', '3', ']', 'Throughout', 'this', 'period', 'he', 'continued', 'to', 'compose', 'and', 'to', 'give', 'recitals', 'in', 'concerts', 'and', 'salons', 'in', 'Warsaw', '.', 'He', 'was', 'engaged', 'by', 'the', 'inventors', 'of', 'a', 'mechanical', 'organ', ',', 'the', '"', 'eolomelodicon', '"', ',', 'and', 'on', 'this', 'instrument', 'in', 'May', '1825', 'he', 'performed', 'his', 'own', 'improvisation', 'and', 'part', 'of', 'a', 'concerto', 'by', 'Moscheles', '.', 'The', 'success', 'of', 'this', 'concert', 'led', 'to', 'an', 'invitation', 'to', 'give', 'a', 'similar', 'recital', 'on', 'the', 'instrument', 'before', 'Tsar', 'Alexander', 'I', ',', 'who', 'was', 'visiting', 'Warsaw', ';', 'the', 'Tsar', 'presented', 'him', 'with', 'a', 'diamond', 'ring', '.', 'At', 'a', 'subsequent', 'eolomelodicon', 'concert', 'on', '10', 'June', '1825', ',', 'Chopin', 'performed', 'his', 'Rondo', 'Op', '.', '1', '.', 'This', 'was', 'the', 'first', 'of', 'his', 'works', 'to', 'be', 'commercially', 'published', 'and', 'earned', 'him', 'his', 'first', 'mention', 'in', 'the', 'foreign', 'press', ',', 'when', 'the', 'Leipzig', 'Allgemeine', 'Musikalische', 'Zeitung', 'praised', 'his', '"', 'wealth', 'of', 'musical', 'ideas', '"', '.'])
        self.assertEqual(tokenize_long_text('The GameCube version was released worldwide in December 2006.[b]'), ['The', 'GameCube', 'version', 'was', 'released', 'worldwide', 'in', 'December', '2006', '.', '[', 'b', ']'])
