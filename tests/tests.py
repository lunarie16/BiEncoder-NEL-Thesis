import unittest

import torch

from texoopy import Dataset, Document, Annotation, NamedEntityAnnotation

import tempfile

from biencodernel.biencoder import CrossEntropyLoss
from biencodernel.datasets import KBDataset, TokenTools
from biencodernel.knn import FaissHNSWIndex, FaissExactKNNIndex


class FaissHNSWIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_vectors = {
            'a': torch.tensor([1.0, 1.0, 1.0]),
            'b': torch.tensor([1.0, 1.0, 0.9]),
            'c': torch.tensor([1.0, 1.0, 0.8]),
            'd': torch.tensor([1.0, 1.0, 0.7]),
            'e': torch.tensor([1.0, 1.0, 0.6]),
            'f': torch.tensor([1.0, 1.0, 0.5]),
            'g': torch.tensor([1.0, 1.0, 0.4]),
            'h': torch.tensor([1.0, 1.0, 0.3])
        }
        self.index = FaissHNSWIndex(self.test_vectors)

    def test_len_k1(self):
        query = self.test_vectors['a']
        self.assertEqual(1, len(self.index.get_knn_ids_for_vector(query, k=1)))

    def test_len_k5(self):
        query = self.test_vectors['a']
        self.assertEqual(5, len(self.index.get_knn_ids_for_vector(query, k=5)))

    def test_len_max(self):
        query = self.test_vectors['a']
        self.assertEqual(8, len(self.index.get_knn_ids_for_vector(query, k=500)))

    def test_self(self):
        query = self.test_vectors['e']
        ids, similarities = zip(*self.index.get_knn_ids_for_vector(query, k=1))
        ids = [id[0] for id in ids]
        self.assertEqual(['e'], ids)

    def test_self_2d_shape(self):
        query = self.test_vectors['e'].view(1, -1)
        ids, similarities = zip(*self.index.get_knn_ids_for_vector(query, k=1))
        ids = [id[0] for id in ids]
        self.assertEqual(['e'], ids)

    def test_neighbours(self):
        query = self.test_vectors['e']
        ids, distances = zip(*self.index.get_knn_ids_for_vector(query, k=3))
        ids = set(ids)
        self.assertEqual({'e', 'd', 'f'}, ids)

    def test_distance_order(self):
        query = self.test_vectors['a']
        _, distances = zip(*self.index.get_knn_ids_for_vector(query, k=3))
        self.assertGreater(distances[1], distances[0])
        self.assertGreater(distances[2], distances[1])

class FaissExactKNNIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_vectors = {
            'a': torch.tensor([1.0, 1.0, 1.0]),
            'b': torch.tensor([1.0, 1.0, 0.9]),
            'c': torch.tensor([1.0, 1.0, 0.8]),
            'd': torch.tensor([1.0, 1.0, 0.7]),
            'e': torch.tensor([1.0, 1.0, 0.6]),
            'f': torch.tensor([1.0, 1.0, 0.5]),
            'g': torch.tensor([1.0, 1.0, 0.4]),
            'h': torch.tensor([1.0, 1.0, 0.3])
        }
        self.index = FaissExactKNNIndex(self.test_vectors)

    def test_len_k1(self):
        query = self.test_vectors['a']
        self.assertEqual(1, len(self.index.get_knn_ids_for_vector(query, k=1)))

    def test_len_k5(self):
        query = self.test_vectors['a']
        self.assertEqual(5, len(self.index.get_knn_ids_for_vector(query, k=5)))

    def test_len_max(self):
        query = self.test_vectors['a']
        self.assertEqual(8, len(self.index.get_knn_ids_for_vector(query, k=500)))

    def test_self(self):
        query = self.test_vectors['e']
        ids, similarities = zip(*self.index.get_knn_ids_for_vector(query, k=1))
        ids = [id[0] for id in ids]
        self.assertEqual(['e'], ids)

    def test_self_2d_shape(self):
        query = self.test_vectors['e'].view(1, -1)
        ids, similarities = zip(*self.index.get_knn_ids_for_vector(query, k=1))
        ids = [id[0] for id in ids]
        self.assertEqual(['e'], ids)

    def test_neighbours(self):
        query = self.test_vectors['e']
        ids, distances = zip(*self.index.get_knn_ids_for_vector(query, k=3))
        ids = set(ids)
        self.assertEqual({'e', 'd', 'f'}, ids)


class CrossEntropyLossFunctionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.loss_mean = CrossEntropyLoss(reduction='mean')
        self.mention_vec1 = torch.tensor([0.30, 0.50, 0.80])
        self.concept_vec1 = torch.tensor([0.25, 0.55, 0.79])
        self.mention_vec2 = torch.tensor([-0.29, 0.51, -0.82])
        self.concept_vec2 = torch.tensor([-0.30, 0.50, -0.80])

    def test_smoke(self):
        mention_vecs = torch.stack([
            self.mention_vec1,
            self.mention_vec2
        ])
        concept_vecs = torch.stack([
            self.concept_vec1,
            self.concept_vec2
        ])
        loss = self.loss_mean(mention_vecs, concept_vecs)
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_good_is_lower_than_bad_entropy(self):
        mention_vecs_good = torch.stack([
            self.mention_vec1,
            self.mention_vec2
        ])
        mention_vecs_bad = torch.stack([
            self.mention_vec2,
            self.mention_vec1
        ])
        concept_vecs = torch.stack([
            self.concept_vec1,
            self.concept_vec2
        ])
        loss_good = self.loss_mean(mention_vecs_good, concept_vecs)
        loss_bad = self.loss_mean(mention_vecs_bad, concept_vecs)
        self.assertLessEqual(loss_good.float(), loss_bad.float())


class KBDatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        self.max_length = 13
        self.concept1_id = 'C001'
        self.concept2_id = 'C002'
        self.concept1_name = 'ConceptName1'
        self.concept2_name = 'ConceptName2'
        self.concept1_text = 'I am concept description number 1!'
        self.concept2_text = 'I am concept description number 2!'
        self.kb_texoo_doc1 = Document(begin=0, end=len(self.concept1_text), id=self.concept1_id,
                                      title=self.concept1_name, text=self.concept1_text)
        self.kb_texoo_doc2 = Document(begin=0, end=len(self.concept2_text), id=self.concept2_id,
                                      title=self.concept2_name, text=self.concept2_text)
        self.kb_texoo_dataset = Dataset(name='TEST_DATASET', language='en')
        self.kb_texoo_dataset.documents.append(self.kb_texoo_doc1)
        self.kb_texoo_dataset.documents.append(self.kb_texoo_doc2)
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(self.kb_texoo_dataset.to_json().encode('utf-8'))
            fp.flush()
            self.kb_dataset_testee = KBDataset(path_to_json=fp.name, max_length=self.max_length)

    def test_item_count(self):
        self.assertEqual(2, len(self.kb_dataset_testee))

    def test_return_correct_concept_id(self):
        concept_id, _ = self.kb_dataset_testee.__getitem__(0)
        self.assertEqual(self.concept1_id, concept_id)

    def test_return_type_tensor(self):
        _, tokens = self.kb_dataset_testee.__getitem__(0)
        self.assertEqual(type(tokens), torch.Tensor)

    def test_token_count(self):
        _, tokens = self.kb_dataset_testee.__getitem__(0)
        self.assertEqual(self.max_length, len(tokens))

    def test_tokenization(self):
        tokenizer = TokenTools.get_tokenizer()
        _, tokens = self.kb_dataset_testee.__getitem__(0)
        self.assertEqual('[CLS] conceptname1 [ent] i am concept description number 1! [SEP]', tokenizer.decode(tokens))
