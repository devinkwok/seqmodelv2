import unittest
from pyfaidx import Fasta
from seqmodel import dataset


class TestSequence(unittest.TestCase):

    def setUp(self):
        self.multiple_fasta = 'test/data/seq/multiple.fasta'
        self.single_fasta = 'test/data/seq/single.fasta'
        self.single_bed = 'test/data/seq/single.bed'

    def test_Alphabet(self):
        tokens = ['a', 'b', 'mask']
        control_tokens = ['none', 'test']
        alphabet = dataset.Alphabet(tokens, control_tokens)
        self.assertEqual(len(alphabet), len(tokens) + len(control_tokens))
        self.assertEqual(alphabet.n_char, len(tokens))
        self.assertEqual(alphabet.none_token, control_tokens[0])

        tokens = ['a', 'a', 'b', 'x', 'mask', 'unknown']
        tgt = ['a', 'a', 'b', 'none', 'mask', 'none']
        ints = [0, 0, 1, 3, 2, 3]
        token2int = alphabet.to_idx(tokens)
        self.assertListEqual(token2int, ints)
        self.assertListEqual(alphabet.to_tokens(token2int), tgt)
        int2token = alphabet.to_tokens(ints)
        self.assertListEqual(int2token, tgt)
        self.assertListEqual(alphabet.to_idx(int2token), ints)

        token_str = alphabet.to_tokens(ints, single_chars=True)
        self.assertEqual(token_str, 'aabnmn')

    def test_FastaSequence(self):
        seq = dataset.FastaSequence(self.multiple_fasta)
        names = ['CIZ78533', 'CCZ78532', 'CFZ78531', 'CMZ78530', 'CLZ78529']
        self.assertListEqual(list(seq.names), names)
        self.assertTrue(seq.exists(names[0], 10, 20))
        self.assertFalse(seq.exists(names[0], 1000, 2000))
        self.assertFalse(seq.exists(names[0], 0, 1000))
        self.assertFalse(seq.exists(names[0], -10, 20))
        self.assertFalse(seq.exists('test', 0, 100))

        ids = [seq.name_to_id(n) for n in names]
        ids2names = [seq.id_to_name(i) for i in ids]
        self.assertListEqual(ids2names, names)
        with self.assertRaises(ValueError):
            seq.id_to_name(123456)

        subseq = seq.get(names[0], 70, 180)
        self.assertEqual(len(subseq), 180 - 70)
        self.assertEqual(subseq,
            'AATCCGGAGGACCGGTGTACTCAGCTCACCGGGGGCATTGCTCCCGTGGTGACCC' + \
            'TGATTTGTTGTTGGGCCGCCTCGGGAGCGTCCATGGCGGGTTTGAACCTCTAGCC')

        alphabet = dataset.DnaAlphabet()
        self.assertEqual(alphabet.to_idx(subseq[:10]),
                        [0, 0, 3, 2, 2, 1, 1, 0, 1, 1])

    def test_Intervals(self):
        fasta = Fasta(self.multiple_fasta)
        intervals = dataset.Intervals.from_fasta_obj(fasta)
        self.assertEqual(len(intervals), len(fasta.keys()))
        for interval, (k, v) in zip(intervals, fasta.items()):
            name, start, end = interval
            self.assertEqual(name, k)
            self.assertEqual(start, 0)
            self.assertEqual(end, len(v))

        intervals = dataset.Intervals.from_bed_file(self.single_bed)


if __name__ == '__main__':
    unittest.main()
