import logging
import os
from enum import Enum
from typing import List, Dict, Tuple, Union
import gc
import numpy as np
import torch
from ray import tune
from torch import nn, Tensor
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertModel
from texoopy import Dataset, MentionAnnotation, NamedEntityAnnotation
from biencodernel.knn import FaissExactKNNIndex

logger = logging.getLogger(__name__)


class PoolingType(Enum):
    CLS = 1
    AVG = 2

    @classmethod
    def from_string(cls, pooling_type: str):
        if pooling_type.lower().strip() == 'avg':
            return cls.AVG
        if pooling_type.lower().strip() == 'cls':
            return cls.CLS
        raise NotImplementedError('Pooling-type "{}" not supported!'.format(pooling_type))

    def __str__(self):
        return self.name


class Encoder(nn.Module):

    def __init__(self, tokenizer: BertTokenizer, freeze_embeddings, bert_model: str = 'bert-base-german-cased'):
        """
        :param tokenizer: The tokenizer that was used to generate the token ids (Necessary for resizing the vocab of the model)
        :param pooling: CLS (CLS token) or AVG
        :param freeze_embeddings: freeze embedding layer as suggested by Humeau et al. 2019
        """

        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        if freeze_embeddings:
            for param in list(self.bert.embeddings.parameters()):
                param.requires_grad = False
        self.bert.resize_token_embeddings(len(tokenizer))

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden_states, cls_tokens = self.bert(token_ids, return_dict=False)
        return cls_tokens


class CrossEntropyLoss:

    def __init__(self, device: str = 'cpu', reduction: str = 'mean'):
        """
        Cross Entropy Loss as mentioned in Humeau et al. 2019 - Poly Encoders
        :param device: cpu | cuda
        """
        logger.info('Initializing CrossEntropyLoss on device {} with reduction {}.'.format(device, reduction))
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss(reduction=reduction).to(self.device)

    def loss(self, mention_vecs: torch.Tensor, concept_vecs: torch.Tensor) -> torch.Tensor:
        assert concept_vecs.size() == mention_vecs.size()
        dot_products = torch.matmul(mention_vecs, concept_vecs.t()).to(self.device)
        y_target = torch.arange(0, concept_vecs.shape[0]).to(self.device)
        return self.loss_func(dot_products, y_target)

    def __call__(self, mention_vecs: torch.Tensor, concept_vecs: torch.Tensor) -> torch.Tensor:
        return self.loss(mention_vecs=mention_vecs, concept_vecs=concept_vecs)


class BiEncoder:

    def __init__(self, device: str, model_path: str, tokenizer: BertTokenizer,
                 freeze_embeddings: bool = True, hpo: bool = False, bert_model: str = 'bert-base-german-cased', writer: SummaryWriter = None):
        self.model_path: str = model_path
        self.device: str = device
        self.tokenizer: BertTokenizer = tokenizer
        self.bert_model = bert_model
        self.encoder_mention: Encoder = Encoder(tokenizer=self.tokenizer, freeze_embeddings=freeze_embeddings, bert_model=self.bert_model).to(self.device)
        self.encoder_concept: Encoder = Encoder(tokenizer=self.tokenizer, freeze_embeddings=freeze_embeddings, bert_model=self.bert_model).to(self.device)
        self.writer: SummaryWriter = writer
        self.loss_func = CrossEntropyLoss(self.device)
        self.do_hpo = hpo

    def train(self,
              train_dataloader: DataLoader,
              dev_dataloader: DataLoader,
              test_dataloader: DataLoader,
              kb_dataloader: DataLoader,
              learning_rate: float,
              valtest_interval: int,
              epochs: int,
              warmup_steps: int,
              omp_num_threads: int,
              do_validation: bool = True,
              do_test: bool = True,
              clip_grad_norm: bool = False,
              ) -> List[float]:

        optimizer_mention = Adam(self.encoder_mention.parameters(), lr=learning_rate)
        optimizer_concept = Adam(self.encoder_concept.parameters(), lr=learning_rate)
        scheduler_mention = get_linear_schedule_with_warmup(optimizer_mention,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=len(train_dataloader) * epochs)
        scheduler_concept = get_linear_schedule_with_warmup(optimizer_concept,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=len(train_dataloader) * epochs)

        gc.collect()
        torch.cuda.empty_cache()
        recallAt1 = []
        for epoch_num in tqdm(range(epochs), desc='Epoch'):
            self.__train_loop(
                dataloader=train_dataloader,
                optimizer_mention=optimizer_mention,
                scheduler_mention=scheduler_mention,
                optimizer_concept=optimizer_concept,
                scheduler_concept=scheduler_concept,
                epoch_num=epoch_num,
                clip_grad_norm=clip_grad_norm
            )

            if valtest_interval != 0 and (epoch_num + 1) % valtest_interval == 0:
                if do_validation:
                    train_loss, dev_loss = self.__validation_loop(
                        dataloader_train=train_dataloader,
                        dataloader_dev=dev_dataloader,
                    )
                    self.writer.add_scalar('avg_epoch_losses/train_loss', np.mean(train_loss), epoch_num)
                    self.writer.add_scalar('avg_epoch_losses/dev_loss', np.mean(dev_loss), epoch_num)
                    self.writer.flush()
                if do_test:
                    dev_results = self.__test_loop(
                        test_dl=dev_dataloader,
                        umls_concept_dl=kb_dataloader,
                        omp_num_threads=omp_num_threads
                    )
                    self.writer.add_scalar('dev/recall_at_1', dev_results['recall@1'], epoch_num)
                    self.writer.add_scalar('dev/recall_at_20', dev_results['recall@20'], epoch_num)
                    self.writer.add_scalar('dev/recall_at_100', dev_results['recall@100'], epoch_num)
                    self.writer.add_scalar('dev/recall_at_1000', dev_results['recall@1000'], epoch_num)
                    self.writer.flush()
                    recallAt1.append(dev_results['recall@1'])
                    if self.do_hpo:
                        tune.report(recallAt1=dev_results['recall@1'], epoch=epoch_num)
                    test_results = self.__test_loop(
                        test_dl=test_dataloader,
                        umls_concept_dl=kb_dataloader,
                        omp_num_threads=omp_num_threads
                    )
                    self.writer.add_scalar('test/recall_at_1', test_results['recall@1'], epoch_num)
                    self.writer.add_scalar('test/recall_at_20', test_results['recall@20'], epoch_num)
                    self.writer.add_scalar('test/recall_at_100', test_results['recall@100'], epoch_num)
                    self.writer.add_scalar('test/recall_at_1000', test_results['recall@1000'], epoch_num)
                    self.writer.flush()

            gc.collect()
            torch.cuda.empty_cache()
        return recallAt1

    def __train_loop(
            self,
            dataloader: DataLoader,
            optimizer_mention: Optimizer,
            scheduler_mention,
            optimizer_concept: Optimizer,
            scheduler_concept,
            epoch_num: int,
            clip_grad_norm: bool,
    ) -> List[float]:
        """
        Performs a training of both encoders and returns the train loss.
        :param dataloader:
        :param optimizer_mention:
        :param optimizer_concept:
        :param epoch_num:
        :return: minibatch losses
        """
        self.encoder_mention.train()
        self.encoder_concept.train()
        minibatch_losses = []
        for step_num, batch_data in tqdm(enumerate(dataloader), desc='Training', total=len(dataloader)):
            mention_tokens, concept_tokens, _ = batch_data
            mention_tokens = mention_tokens.to(self.device)
            concept_tokens = concept_tokens.to(self.device)
            vecs_mention = self.encoder_mention(mention_tokens)
            vecs_concept = self.encoder_concept(concept_tokens)
            batch_loss = self.loss_func(vecs_mention, vecs_concept)
            self.writer.add_scalar('batch_losses/train_loss'.format(epoch_num), batch_loss.item(), step_num)
            self.writer.flush()
            minibatch_losses.append(batch_loss.item())
            self.encoder_mention.zero_grad()
            self.encoder_concept.zero_grad()
            batch_loss.backward()
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.encoder_mention.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.encoder_concept.parameters(), 1.0)
            optimizer_mention.step()
            optimizer_concept.step()
            scheduler_mention.step()
            scheduler_concept.step()
            gc.collect()
            torch.cuda.empty_cache()
        return minibatch_losses

    def __validation_loop(self,
                          dataloader_train: DataLoader,
                          dataloader_dev: DataLoader) -> Tuple[List[float], List[float]]:
        """
        Performs a Evaluation on the dev set of both encoders and returns the dev loss.
        :param dataloader_train:
        :param dataloader_dev:
        :return: Tuple of minibatch losses for train and dev
        """
        with torch.no_grad():
            self.encoder_mention.eval()
            self.encoder_concept.eval()
            minibatch_losses_train = []
            minibatch_losses_dev = []
            for step_num, batch_data in tqdm(enumerate(dataloader_train), desc='Eval on training set',
                                             total=len(dataloader_train)):
                mention_tokens, concept_tokens, _ = batch_data
                mention_tokens = mention_tokens.to(self.device)
                concept_tokens = concept_tokens.to(self.device)
                vecs_mention = self.encoder_mention(mention_tokens)
                vecs_concept = self.encoder_concept(concept_tokens)
                batch_loss = self.loss_func(vecs_mention, vecs_concept)
                minibatch_losses_train.append(batch_loss.item())
            for step_num, batch_data in tqdm(enumerate(dataloader_dev), desc='Evaluation on dev set',
                                             total=len(dataloader_dev)):
                mention_tokens, concept_tokens, _ = batch_data
                mention_tokens = mention_tokens.to(self.device)
                concept_tokens = concept_tokens.to(self.device)
                vecs_mention = self.encoder_mention(mention_tokens)
                vecs_concept = self.encoder_concept(concept_tokens)
                batch_loss = self.loss_func(vecs_mention, vecs_concept)
                minibatch_losses_dev.append(batch_loss.item())
            return minibatch_losses_train, minibatch_losses_dev

    def __test_loop(self, test_dl: DataLoader, umls_concept_dl: DataLoader, omp_num_threads: int) -> Dict[str, float]:

        with torch.no_grad():
            self.encoder_mention.eval()
            self.encoder_concept.eval()

            concept_embeddings_cache = dict()
            for step_num, batch_data in tqdm(enumerate(umls_concept_dl), desc='Generating KB candidate embeddings',
                                             total=len(umls_concept_dl)):
                concept_ids, concept_tokens = batch_data
                concept_tokens = concept_tokens.to(self.device)
                concept_embeddings = self.encoder_concept(concept_tokens)
                for cui, concept_embedding in zip(concept_ids, concept_embeddings):
                    concept_embeddings_cache[cui] = concept_embedding.to('cpu')
            knn_index = FaissExactKNNIndex(concept_embeddings_cache, normalize=True)
            del concept_embeddings_cache

            total_examples = 0
            positive_at_1 = 0
            positive_at_20 = 0
            positive_at_100 = 0
            positive_at_1000 = 0

            for step_num, batch_data in tqdm(enumerate(test_dl), desc='Test', total=len(test_dl)):
                mention_tokens, concept_tokens, concept_ids = batch_data
                mention_tokens = mention_tokens.to(self.device)
                mention_embeddings = self.encoder_mention(mention_tokens)
                for cui_gold, mention_embedding in zip(concept_ids, mention_embeddings.to('cpu')):
                    total_examples += 1
                    knn_ids, _ = zip(*knn_index.get_knn_ids_for_vector(mention_embedding, k=1000))

                    if cui_gold in knn_ids[:1]:
                        positive_at_1 += 1
                    if cui_gold in knn_ids[:20]:
                        positive_at_20 += 1
                    if cui_gold in knn_ids[:100]:
                        positive_at_100 += 1
                    if cui_gold in knn_ids[:1000]:
                        positive_at_1000 += 1

            results = {
                'recall@1': positive_at_1 / total_examples,
                'recall@20': positive_at_20 / total_examples,
                'recall@100': positive_at_100 / total_examples,
                'recall@1000': positive_at_1000 / total_examples
            }

            return results

    def save_encoders(self, path, experiment_descriptor):
        # TODO the tokenizers should be saved as well in this directory
        path = os.path.join(path, experiment_descriptor)
        os.makedirs(path, exist_ok=True)
        mention_encoder_path = os.path.join(path, 'encoder_mention.statedict')
        concept_encoder_path = os.path.join(path, 'encoder_concept.statedict')
        torch.save(self.encoder_mention.state_dict(), mention_encoder_path)
        torch.save(self.encoder_concept.state_dict(), concept_encoder_path)

    def save_checkpoint(self, path, experiment_descriptor, epoch):
        # TODO store both encoders as well as the state of optimizers and epoch.
        # We will need this if we want to implement the resuming of trainings after the pod crashed.
        raise NotImplementedError
