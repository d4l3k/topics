import unittest

import vectors

class TestVectors(unittest.TestCase):

    def test_is_stopword(self):
        self.assertTrue(vectors.is_stopword('the'))
        self.assertTrue(vectors.is_stopword('.'))
        self.assertFalse(vectors.is_stopword('cow'))
