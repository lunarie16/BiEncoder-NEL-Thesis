import logging
import math
import os
import pickle
from enum import Enum
from typing import List, Iterable, Tuple, Union
import codecs
import flair
import torch
import ujson as json
from flair.data import Sentence, Span
from flair.models import SequenceTagger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from texoopy import Document, Annotation, Dataset
from biencodernel.biencoder import Encoder
from biencodernel.knn import FaissExactKNNIndex
import logging

logger = logging.getLogger(__name__)


class MentionTokenizer:

    def __init__(self, max_length: int, bert_model: str = 'bert-base-german-cased'):
        self.logger = logging.getLogger(__name__)
        self.max_length = max_length
        self.bert_model = bert_model
        self.bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case='uncased' in self.bert_model)
        self.bert_tokenizer.add_special_tokens({'additional_special_tokens': ['[ent]', '[ms]', '[me]']})

    def __tokenize_crop_pad(self, text: str) -> List[int]:
        """
        Returns a a centered version of the mention tokens, cropped/padded to max_length
        :param text:
        :return:
        """

        def center_mention(tokens: List[str], max_length: int, ms_token: str = '[ms]', me_token: str = '[me]'):
            """
            Centers and crops the mention based on the position of the boundary tokens and adds [CLS] and [SEP]
            :param tokens: tokens (without [CLS] and [SEP])
            :param max_length: maximal length of the final output (no padding)
            :param ms_token: start token of mention
            :param me_token: end token of mention
            :return: List of tokens
            """

            idx_ms, idx_me = tokens.index('[ms]'), tokens.index('[me]')
            num_tokens_mention = idx_me - idx_ms + 1
            max_context_length = int(math.ceil((max_length - num_tokens_mention) / 2 - 1))
            tokens_mention = tokens[idx_ms:idx_me + 1]
            tokens_left = tokens[max(0, idx_ms - max_context_length):idx_ms]
            tokens_right = tokens[idx_me + 1:min(len(tokens), idx_me + 1 + max_context_length)]
            tokens_cropped = ['[CLS]'] + tokens_left + tokens_mention + tokens_right
            return tokens_cropped[:max_length - 1] + ['[SEP]']

        def pad(token_ids: List[int], max_length: int) -> List[int]:
            if len(token_ids) < self.max_length:
                token_ids = token_ids + [0] * (max_length - len(token_ids))
            return token_ids

        tokens = self.bert_tokenizer.tokenize(text)
        tokens_cropped = center_mention(tokens, self.max_length)
        token_ids_cropped = self.bert_tokenizer.convert_tokens_to_ids(tokens_cropped)
        return pad(token_ids_cropped, self.max_length)

    def tokenize_flair_span(self, document: Document, mention: Span) -> torch.Tensor:
        mention_with_context = '{} [ms] {} [me] {}'.format(
            document.get('text')[0:mention.start_pos],
            document.get('text')[mention.start_pos:mention.end_pos],
            document.get('text')[mention.end_pos:]
        )
        return torch.tensor(self.__tokenize_crop_pad(mention_with_context))

    def bert_tokenize_kb_concepts(self, concepts: List) -> List[Tuple[str, List[int]]]:
        """
        :param concepts: KB Concepts
        :return: A list of tuples (CUI, TokenIds)
        """
        results = []
        for concept in tqdm(concepts, desc='Tokenizing KB concepts'):
            kb_id = concept['id']
            name = concept['title']
            description = concept['text']
            tokens = self.tokenizer.encode(name + ' [ent] ' + description, max_length=self.max_length,
                                           pad_to_max_length=True)
            results.append((kb_id, tokens))
        return results


class NerNelEvaluation:

    class IndexType(Enum):
        FAISSExactNormalized = 1

    def __init__(
            self,
            test_path: str,
            ner_model_path: str,
            encoder_path: str,
            precomputed_concept_embeddings_path: Union[str, None],
            kb_path: str,
            encoder_input_length: int,
            index_type: IndexType,
            bert_model: str,
            ignore_cache: bool,
            device: str = 'cuda'):
        self.precomputed_concept_embeddings_path = precomputed_concept_embeddings_path
        self.logger = logging.getLogger(__name__)
        self.encoder_path = encoder_path
        self.device = device
        self.bert_model = bert_model
        flair.device = torch.device(self.device)
        self.kb_path: str = kb_path
        f = codecs.open(test_path, 'r', 'utf-8-sig')
        self.test = json.load(f)
        f.close()
        self.ner: SequenceTagger = SequenceTagger.load(ner_model_path)
        self.tokenizer: MentionTokenizer = MentionTokenizer(max_length=encoder_input_length, bert_model=self.bert_model)

        self.mention_encoder = Encoder(
            tokenizer=self.tokenizer.bert_tokenizer, freeze_embeddings=True,  bert_model=self.bert_model).to(self.device)
        self.mention_encoder.load_state_dict(
            torch.load(os.path.join(encoder_path + '/encoder_mention.statedict'), map_location=self.device))
        self.mention_encoder.eval()

        self.concept_index = self.__build_concept_index(
            index_type=index_type,
            ignore_cache=ignore_cache)

    def __build_concept_index(self, index_type: IndexType, ignore_cache: bool):
        kb_concept_embeddings_cache = dict()
        with open(self.precomputed_concept_embeddings_path, 'r') as f_pe:
            for line in tqdm(f_pe, desc='Reading pre-computed vectors'):
                obj = json.loads(line)
                kb_id = obj['id']
                vec = torch.tensor(obj['vec'])
                kb_concept_embeddings_cache[kb_id] = vec

        if index_type == NerNelEvaluation.IndexType.FAISSExactNormalized:
            self.logger.info('Building FaissExactKNNIndex')
            return FaissExactKNNIndex(kb_concept_embeddings_cache, normalize=True)

    def link_mention(self, document: Document, mention: Span) -> Tuple[str, float]:
        """
        :param document:
        :param span:
        :return: CUI
        """
        tokens = self.tokenizer.tokenize_flair_span(document=document, mention=mention)
        with torch.no_grad():
            mention_embedding = self.mention_encoder(tokens.to(self.device).view(1, -1))
        return self.concept_index.get_knn_ids_for_vector(mention_embedding, k=1)[0]

    def __m_w(self, p1: int, l1: int, ent1: str, p2: int, l2: int, ent2: str) -> bool:
        # For  a better understanding of the variable names below, look at
        # Cornolti et al. 2013 - "A Framework for Benchmarking Entity-Annotation Systems".
        e1 = p1 + l1 - 1
        e2 = p2 + l2 - 1
        if p1 <= p2 <= e1 or p1 <= e2 <= e1 or p2 <= p1 <= e2 or p2 <= e1 <= e2:
            return ent1 == ent2
        return False

    def run_micro(self) -> Tuple[int, int, int]:
        """
        Performs evaluation according to init params.
        :return: (tp, fp, fn, tn)
        """

        tp = 0
        fp = 0
        fn = 0

        document: Document
        for document in tqdm(self.test['documents'], desc='Evaluating'):
            flair_sentence: Sentence = Sentence(document['text'], use_tokenizer=True)
            self.ner.predict(flair_sentence)
            s = flair_sentence.get_spans('BIOES')

            s_linked_concepts_cache = []

            for elem_s in s:
                s_ent, _ = self.link_mention(document, elem_s)
                s_linked_concepts_cache.append((elem_s, s_ent))

            logger.info(f'this is a s_linked_concepts_cache {s_linked_concepts_cache}')

            elem_s: Span
            for elem_s, s_ent in s_linked_concepts_cache:
                s_p = elem_s.start_pos
                s_l = elem_s.end_pos - s_p
                elem_g: Annotation
                found_match_in_g: bool = False
                for elem_g in document.get('annotations'):
                    g_p = elem_g.get('begin')
                    g_l = elem_g.get('length')
                    g_ent = elem_g.get('refId')
                    if self.__m_w(s_p, s_l, s_ent, g_p, g_l, g_ent):
                        # there exists an element in s that has a matching element in g
                        tp += 1
                        found_match_in_g = True
                        break
                if not found_match_in_g:
                    logger.info(f'not found in match g {elem_s, s_ent, s_p, s_l}')
                    # there exists an element in s that has no matching element in g
                    fp += 1

            for elem_g in document.get('annotations'):
                g_p = elem_g.get('begin')
                g_l = elem_g.get('length')
                g_ent = elem_g.get('refId')
                found_match_in_s = False
                for elem_s, s_ent in s_linked_concepts_cache:
                    s_p = elem_s.start_pos
                    s_l = elem_s.end_pos - s_p
                    if self.__m_w(s_p, s_l, s_ent, g_p, g_l, g_ent):
                        found_match_in_s = True
                        break
                if not found_match_in_s:
                    fn += 1

        return tp, fp, fn

    def run_macro(self) -> dict:
        """
        Performs evaluation according to init params.
        :return: (tp, fp, fn, tn)
        """
        macro = {}
        id_with_types = {}
        kb = json.load(codecs.open(self.kb_path, 'r', 'utf-8-sig'))
        for entity in kb['documents']:
            if not entity['type'] in macro:
                macro[entity['type']] = {
                    'tp': 0,
                    'fp': 0,
                    'fn': 0
                }
            id_with_types[entity['id']] = entity['type']

        document: Document
        for document in tqdm(self.test['documents'], desc='Evaluating'):
            flair_sentence: Sentence = Sentence(document['text'], use_tokenizer=True)
            self.ner.predict(flair_sentence)
            s = flair_sentence.get_spans('BIOES')

            s_linked_concepts_cache = []

            for elem_s in s:
                s_ent, _ = self.link_mention(document, elem_s)
                s_linked_concepts_cache.append((elem_s, s_ent))

            logger.info(f'this is a s_linked_concepts_cache {s_linked_concepts_cache}')

            elem_s: Span
            for elem_s, s_ent in s_linked_concepts_cache:
                s_p = elem_s.start_pos
                s_l = elem_s.end_pos - s_p
                elem_g: Annotation
                found_match_in_g: bool = False
                for elem_g in document.get('annotations'):
                    g_p = elem_g.get('begin')
                    g_l = elem_g.get('length')
                    g_ent = elem_g.get('refId')
                    g_type = id_with_types[g_ent]

                    if self.__m_w(s_p, s_l, s_ent, g_p, g_l, g_ent):
                        # there exists an element in s that has a matching element in g
                        macro[g_type]['tp'] =  macro[g_type]['tp'] + 1
                        found_match_in_g = True
                        break
                if not found_match_in_g:
                    logger.info(f'not found in match g {elem_s, s_ent, s_p, s_l}')
                    # there exists an element in s that has no matching element in g
                    macro[g_type]['fp'] = macro[g_type]['fp'] + 1

            for elem_g in document.get('annotations'):
                g_p = elem_g.get('begin')
                g_l = elem_g.get('length')
                g_ent = elem_g.get('refId')
                g_type = id_with_types[g_ent]
                found_match_in_s = False
                for elem_s, s_ent in s_linked_concepts_cache:
                    s_p = elem_s.start_pos
                    s_l = elem_s.end_pos - s_p
                    if self.__m_w(s_p, s_l, s_ent, g_p, g_l, g_ent):
                        found_match_in_s = True
                        break
                if not found_match_in_s:
                    macro[g_type]['fn'] = macro[g_type]['fn'] + 1

        return macro


