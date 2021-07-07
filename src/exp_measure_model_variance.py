import logging
from typing import List, Dict

import torch
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import get_linear_schedule_with_warmup, BertTokenizer

from biencodernel.biencoder import Encoder, CrossEntropyLoss
from biencodernel.datasets import KBDataset, MentionDataset
from biencodernel.knn import FaissExactKNNIndex
from biencodernel.utils import get_config_from_env

logger = logging.getLogger(__name__)


class BiEncoderVarianceTest:

    def __init__(self, device: str, tokenizer: BertTokenizer, freeze_embeddings: bool):
        self.device = device
        self.tokenizer = tokenizer
        self.encoder_mention = Encoder(tokenizer=self.tokenizer, freeze_embeddings=freeze_embeddings).to(self.device)
        self.encoder_concept = Encoder(tokenizer=self.tokenizer, freeze_embeddings=freeze_embeddings).to(self.device)
        self.loss_func = CrossEntropyLoss(self.device)

    def train(self,
              train_dataloader: DataLoader,
              test_dataloader: DataLoader,
              kb_dataloader: DataLoader,
              learning_rate: float,
              epochs: int,
              warmup_steps: int,
              omp_num_threads: int,
              ) -> Dict[str, float]:

        optimizer_mention = Adam(self.encoder_mention.parameters(), lr=learning_rate)
        optimizer_concept = Adam(self.encoder_concept.parameters(), lr=learning_rate)
        scheduler_mention = get_linear_schedule_with_warmup(optimizer_mention,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=len(train_dataloader) * epochs)
        scheduler_concept = get_linear_schedule_with_warmup(optimizer_concept,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=len(train_dataloader) * epochs)

        for epoch_num in range(epochs):
            self.__train_loop(
                dataloader=train_dataloader,
                optimizer_mention=optimizer_mention,
                scheduler_mention=scheduler_mention,
                optimizer_concept=optimizer_concept,
                scheduler_concept=scheduler_concept,
            )

        return self.__test_loop(
            test_dl=test_dataloader,
            umls_concept_dl=kb_dataloader,
            omp_num_threads=omp_num_threads
        )

    def __train_loop(
            self,
            dataloader: DataLoader,
            optimizer_mention: Optimizer,
            scheduler_mention,
            optimizer_concept: Optimizer,
            scheduler_concept,
    ) -> List[float]:
        """
        Performs a training of both encoders and returns the train loss.
        :param dataloader:
        :param optimizer_mention:
        :param optimizer_concept:
        :return: minibatch losses
        """
        self.encoder_mention.train()
        self.encoder_concept.train()
        minibatch_losses = []
        for step_num, batch_data in enumerate(dataloader):
            mention_tokens, concept_tokens, _ = batch_data
            mention_tokens = mention_tokens.to(self.device)
            concept_tokens = concept_tokens.to(self.device)
            vecs_mention = self.encoder_mention(mention_tokens)
            vecs_concept = self.encoder_concept(concept_tokens)
            batch_loss = self.loss_func(vecs_mention, vecs_concept)
            minibatch_losses.append(batch_loss.item())
            self.encoder_mention.zero_grad()
            self.encoder_concept.zero_grad()
            batch_loss.backward()
            optimizer_mention.step()
            optimizer_concept.step()
            scheduler_mention.step()
            scheduler_concept.step()

        return minibatch_losses

    def __test_loop(self, test_dl: DataLoader, umls_concept_dl: DataLoader, omp_num_threads: int) -> Dict[str, float]:

        with torch.no_grad():
            self.encoder_mention.eval()
            self.encoder_concept.eval()

            umls_concept_embeddings_cache = dict()
            for step_num, batch_data in enumerate(umls_concept_dl):
                concept_ids, concept_tokens = batch_data
                concept_tokens = concept_tokens.to(self.device)
                concept_embeddings = self.encoder_concept(concept_tokens)
                for cui, concept_embedding in zip(concept_ids, concept_embeddings):
                    umls_concept_embeddings_cache[cui] = concept_embedding.to('cpu')

            knn_index = FaissExactKNNIndex(umls_concept_embeddings_cache, omp_num_threads=omp_num_threads)
            del umls_concept_embeddings_cache

            total_examples = 0
            positive_at_1 = 0
            positive_at_20 = 0
            positive_at_100 = 0
            positive_at_1000 = 0

            for step_num, batch_data in enumerate(test_dl):
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


if __name__ == '__main__':

    config = get_config_from_env()

    kb_ds = KBDataset(path_to_json=config['paths']['kb'], max_length=config['input_length'])
    train_ds = MentionDataset(path_to_json=config['paths']['train'], max_length=config['input_length'],
                              kb_dataset=kb_ds)
    test_ds = MentionDataset(path_to_json=config['paths']['test'], max_length=config['input_length'], kb_dataset=kb_ds)

    kb_dl = DataLoader(dataset=kb_ds, batch_size=config['batch_size'], drop_last=True, pin_memory=True)
    train_dl = DataLoader(dataset=train_ds, sampler=RandomSampler(train_ds), batch_size=config['batch_size'],
                          drop_last=True, pin_memory=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=config['batch_size'], drop_last=True, pin_memory=True)

    for warmup_steps in [0, 100, 1000, 10000]:

        for run in range(10):
            biencoder = BiEncoderVarianceTest(
                device=config['device'],
                tokenizer=train_ds.tokenizer,
                freeze_embeddings=config['freeze_emb']
            )

            test_results = biencoder.train(
                train_dataloader=train_dl,
                test_dataloader=test_dl,
                kb_dataloader=kb_dl,
                learning_rate=config['learning_rate'],
                warmup_steps=warmup_steps,
                epochs=config['epochs'],
                omp_num_threads=config['omp_num_threads']
            )
            print('warmup steps', warmup_steps, 'run', run, test_results)
