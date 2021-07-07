import json
import logging
import os
import tempfile
from io import StringIO
import codecs
from tempfile import _TemporaryFileWrapper
import gc
import flair
import torch
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, WordEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from texoopy import Dataset, Document, Annotation, MentionAnnotation
from tqdm import tqdm
import ray
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter, SequenceTaggerParamSelector, OptimizationValue

logger = logging.getLogger('__name__')


def _teXooDocument2FlairBIOESSentence(document: Document, apply_sentence_split: bool = False) -> Sentence:
    if apply_sentence_split:
        raise NotImplemented()  # TODO implement me

    flair_sentence: Sentence = Sentence(document.text, use_tokenizer=True)
    for token in flair_sentence.tokens:
        token.add_tag('BIOES', 'O-ENT')
    annotation: Annotation
    for ann in document.annotations:
        begin = ann.begin
        end = ann.begin + ann.length
        tokens = list()
        for token in flair_sentence.tokens:
            if token.start_pos >= begin and token.end_pos <= end:
                tokens.append(token)
        if len(tokens) == 1:
            if tokens[0].get_tag('BIOES').value != 'O-ENT':
                continue
            tokens[0].add_tag('BIOES', 'S-ENT')
        elif len(tokens) > 1:
            # just make sure that tokens are sorted properly
            tokens = sorted(tokens, key=lambda tok: tok.start_pos)
            existing_tags = set([tok.get_tag('BIOES').value for tok in tokens])
            if existing_tags != {'O-ENT'}:
                # Some tokens are already tagged, skip annotation
                continue
            for tok in tokens:
                tok.add_tag('BIOES', 'I-ENT')
            tokens[0].add_tag('BIOES', 'B-ENT')
            tokens[-1].add_tag('BIOES', 'E-ENT')
    return flair_sentence


def teXoo2CoNLLFile(path_to_json: str) -> _TemporaryFileWrapper:
    """
    Reads a TeXoo JSON, generates a temporary CoNLL-style NER training file
    :param path_to_json:
    :return:
    """
    with codecs.open(path_to_json, 'r', 'utf-8-sig') as f:
        dataset: Dataset = Dataset.from_json(json.load(f))
    document: Document
    f: _TemporaryFileWrapper = tempfile.NamedTemporaryFile()
    for document in tqdm(dataset.documents, desc='Preprocessing {} for NER training'.format(path_to_json)):
        flair_sentence = _teXooDocument2FlairBIOESSentence(document)
        for token in flair_sentence.tokens:
            line: str = '{} \t {}\n'.format(token.text, token.get_tag('BIOES').value)
            f.write(line.encode('utf-8'))
        f.write(b'\n')
    logger.info('Parsed TeXoo dataset {} to temporary CoNLL file {}'.format(path_to_json, f.name))
    return f


class NER:
    def __init__(self, model_base_path: str, tensorboard_path: str, device: str = 'cpu', max_epochs: int = 150,
                 batch_size: int = 8, learning_rate: float = 0.1, search_space: SearchSpace = None):
        """
        Initialises an abstraction layer over the Flair SequenceTagger
        :param device:
        :param model_base_path: Path to model base folder
        """
        self.device = device
        self.is_trained: bool = False
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.model_path = os.path.join(model_base_path, 'ner')
        self.learning_rate = learning_rate
        self.tensorboard_path = tensorboard_path
        self.search_space = search_space
        flair.device = torch.device(device)
        try:
            self.ner: SequenceTagger = SequenceTagger.load(os.path.join(self.model_path, 'best-model.pt'))
            self.is_trained = False
        except FileNotFoundError:
            logger.info('No NER model found, needs to be trained.')

    def train(self, train_texoo: str, test_texoo: str, dev_texoo: str):
        """
        :param train_texoo: Path to train TeXoo file
        :param test_texoo: Path to test TeXoo file
        :param dev_texoo: Path to dev TeXoo file
        :return: 
        """
        if self.is_trained:
            logger.warning('Re-training a previously trained model...')
        f_train = teXoo2CoNLLFile(train_texoo)
        f_test = teXoo2CoNLLFile(test_texoo)
        f_dev = teXoo2CoNLLFile(dev_texoo)
        corpus: Corpus = ColumnCorpus('.', {0: 'text', 1: 'BIOES'},
                                      train_file=f_train.name,
                                      test_file=f_test.name,
                                      dev_file=f_dev.name)
        tag_type = 'BIOES'
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        embedding_types = [
            WordEmbeddings('de'),  # FastText embedding
            FlairEmbeddings('de-forward'),
            FlairEmbeddings('de-backward'),
        ]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
        if self.search_space:
            self.search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings])
            self.search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[64, 128, 256])
            param_selector = SequenceTaggerParamSelector(
                corpus,
                tag_type=tag_type,
                base_path=self.model_path,
                max_epochs=100,
                training_runs=1,
                optimization_value=OptimizationValue.DEV_SCORE
            )
            param_selector.optimize(self.search_space, max_evals=10)
        else:
            from flair.models import SequenceTagger
            tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                    embeddings=embeddings,
                                                    tag_dictionary=tag_dictionary,
                                                    tag_type=tag_type)

            from flair.trainers import ModelTrainer
            trainer: ModelTrainer = ModelTrainer(tagger, corpus,
                                                 use_tensorboard=True)
            trainer.train(base_path=self.model_path,
                          learning_rate=self.learning_rate,
                          mini_batch_size=self.batch_size,
                          max_epochs=self.max_epochs,
                          embeddings_storage_mode=None
                          )

        gc.collect()
        torch.cuda.empty_cache()
        self.is_trained = True
        f_train.close()
        f_test.close()
        f_dev.close()

    def predict(self, dataset: Dataset) -> None:
        """
        Applies NER model onto the given dataset and creates MentionAnnotations of type PREDICT with confidence score.
        :param dataset:
        :return:
        """
        for doc in tqdm(dataset.documents, desc='Applying pre-trained NER model'):
            flair_doc = _teXooDocument2FlairBIOESSentence(doc)
            self.ner.predict(flair_doc)
            for entity in flair_doc.get_spans('BIOES'):
                doc.annotations.append(MentionAnnotation(
                    begin=entity.start_pos,
                    length=entity.end_pos - entity.start_pos,
                    text=entity.text,
                    source='PRED',
                    confidence=entity.score
                ))
