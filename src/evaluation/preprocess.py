import logging
import math
import os
from enum import Enum
from typing import Iterable, List, Tuple

from texoopy import Document, Annotation
from tqdm import tqdm
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class TokenizationMode(Enum):
    REGULAR = 1
    NO_MENTION_SPECIAL_TOKEN = 2
    MASKED_MENTION = 3
    FTM = 4


class Preprocesor:
    def __init__(self, model_path: str,  bert_model: str, max_length: int = 512):
        """
        Initializes the preprocessor for BERT tokenization of ServiceTickets documents and KB concepts.
        Because the tokenizer uses custom tokens, it needs to be saved.
        :param model_path: Path to save the tokenizer
        :param max_length: pad / trim the tokens to this length
        """
        logger.info('Initializing preprocessor ...')
        logger.info('model_path: {}, max_length {}'.format(
            model_path,
            max_length,
        ))
        self.max_length = max_length
        self.ftm_mention_length = max_length  # statically set ftm max_length to max_length
        self.bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case='uncased' in self.bert_model)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ent]', '[ms]', '[me]']})
        os.makedirs(model_path, exist_ok=True)
        self.tokenizer.save_pretrained(model_path)
        # if self.tokenizer.encode('[ENT] [ent] [MS] [ms] [ME] [me]') != [101, 30522, 30522, 30523, 30523, 30524, 30524,
        #                                                                 102]:
        #     logger.error('Unexpected special token ids, check tokenizer here!')
        #     exit(1)

    def __filter_out_special_tokens(self, text: str):
        """
        Replaces every [ or ] with < or >, this takes care of special tokens that are part of the text.
        :return: text
        """
        return text.replace('[', '<').replace(']', '>')

    def __bert_tokenize_annotation(
            self,
            annotation: Annotation,
            document: Document,
            mode: TokenizationMode = TokenizationMode.REGULAR
    ) -> Tuple[str, List[int]]:

        def pad(token_ids: List[int], max_length: int) -> List[int]:
            if len(token_ids) < self.max_length:
                token_ids = token_ids + [0] * (max_length - len(token_ids))
            return token_ids

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

        # TODO: not sure if this is right?!
        cui = annotation.refId
        doc_text = self.__filter_out_special_tokens(document.get_text())
        if mode == TokenizationMode.REGULAR or mode == TokenizationMode.NO_MENTION_SPECIAL_TOKENS:
            mention_with_context = '{}[ms] {} [me]{}'.format(
                doc_text[0:annotation.begin],
                doc_text[annotation.begin:annotation.end],
                doc_text[annotation.end:]
            )
        elif mode == TokenizationMode.MASKED_MENTION:
            mention_with_context = '{}[ms] {} [me]{}'.format(
                doc_text[0:annotation.begin],
                '[MASK]',
                doc_text[annotation.end:]
            )
        elif mode == TokenizationMode.FTM:
            mention_with_context = '[ms] {} [me]'.format(
                doc_text[annotation.begin:annotation.end]
            )
        else:
            raise ValueError('{} is not a valid tokenization mode!'.format(mode))

        tokens = self.tokenizer.tokenize(mention_with_context)
        if mode == TokenizationMode.NO_MENTION_SPECIAL_TOKENS:
            tokens_cropped = center_mention(tokens, self.max_length + 2)
            tokens_cropped.remove('[ms]')
            try:
                tokens_cropped.remove('[me]')
            except ValueError:
                logger.info('Tried to remove [me] token but there was none. It seems that max_length < mention length.')
                tokens_cropped.pop(
                    len(tokens_cropped) - 2)  # remove 2nd last token to fit limit and to keep the [SEP] token.
        elif mode == TokenizationMode.FTM:
            tokens_cropped = center_mention(tokens, self.ftm_mention_length)
        else:
            tokens_cropped = center_mention(tokens, self.max_length)

        token_ids_cropped = self.tokenizer.convert_tokens_to_ids(tokens_cropped)
        if mode != TokenizationMode.FTM:
            token_ids_cropped_padded = pad(token_ids_cropped, self.max_length)
        else:
            token_ids_cropped_padded = pad(token_ids_cropped, self.ftm_mention_length)
        return cui, token_ids_cropped_padded

    def bert_tokenize_documents(self, docs: Iterable[Document], mode: TokenizationMode) -> List[
        Tuple[str, List[int]]]:
        """
        Returns a list of all tokenized annotations from the documents as well as their annotated CUI
        :param mode: Mode of tokenization (regular, no mention special tokens, masked mention)
        :param docs: Iterable of documents
        :return: a list of tuples (cui, tokens), one per annotation
        """
        tokenized_mentions = []
        for doc in tqdm(docs, desc='Tokenizing documents'):
            for annotation in doc.annotations:
                tokenized_mentions.append(self.__bert_tokenize_annotation(annotation, doc, mode))
        return tokenized_mentions

    def bert_tokenize_documents_ftm(self, docs: Iterable[Document]) -> List[Tuple[str, List[int]]]:
        """
        ONLY MENTION, NO CONTEXT! Returns a list of all tokenized annotations from the documents as well as their annotated CUI
        :param docs: Iterable of  documents
        :return: a list of tuples (cui, tokens), one per annotation, token_ids unpadded
        """
        tokenized_mentions = []
        for doc in tqdm(docs, desc='Tokenizing documents'):
            for annotation in doc.annotations:
                cui, token_ids = self.__bert_tokenize_annotation(annotation, doc, mode=TokenizationMode.FTM)
                tokenized_mentions.append((cui, token_ids))

        return tokenized_mentions

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
                                           pad_to_max_length=True, truncation=True)
            results.append((kb_id, tokens))
        return results
