import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from biencodernel.biencoder import Encoder
from biencodernel.utils import StringIdTensorDataset
from evaluation.preprocess import Preprocesor
import codecs
from typing import List


class ConceptEmbGenerator:
    def __init__(self, model_descriptor: str,
                 model_path: str,
                 trained_concept_encoder_path: str,
                 kb_path: str,
                 jsonlines_out_path: str,
                 input_length: int,
                 batch_size: int,
                 device: str,
                 bert_model: str,
                 freeze_embeddings: bool = True
                 ):
        self.kb_path = kb_path
        self.model_descriptor = model_descriptor
        self.model_path = model_path
        self.trained_concept_encoder_path = trained_concept_encoder_path
        self.kb_path = kb_path
        self.jsonlines_out_path = jsonlines_out_path
        self.input_length = input_length
        self.batch_size = batch_size
        self.device = device
        self.freeze_embeddings = freeze_embeddings
        self.bert_model = bert_model

    def generate(self, trained: bool = True):
        preproc = Preprocesor(model_path=self.model_path, bert_model=self.bert_model, max_length=self.input_length)
        f = codecs.open(self.kb_path, 'r', 'utf-8-sig')
        kb = json.load(f)
        f.close()

        concept_encoder = Encoder(tokenizer=preproc.tokenizer,
                                  freeze_embeddings=self.freeze_embeddings, bert_model=self.bert_model)
        if trained:
            concept_encoder.load_state_dict(torch.load(self.trained_concept_encoder_path, map_location=self.device))
        concept_encoder = concept_encoder.to(self.device)
        concept_encoder.eval()

        kb_concepts_to_tokenize: List = kb['documents']
        kb_concepts_tokenized = preproc.bert_tokenize_kb_concepts(kb_concepts_to_tokenize)
        kb_token_tensors = []
        for kb_id, concept_tokens in kb_concepts_tokenized:
            kb_token_tensors.append((kb_id, torch.tensor(concept_tokens)))

        # TODO: there was drop_last=False, why? In train_default it is set to True
        kb_dataloader: DataLoader = DataLoader(StringIdTensorDataset(kb_token_tensors), batch_size=self.batch_size,
                                               pin_memory=True, drop_last=True)
        del kb_token_tensors
        with open(self.jsonlines_out_path, 'w') as f_out:
            with torch.no_grad():
                for batch_data in tqdm(kb_dataloader, desc='Generating Kb concept embeddings'):
                    kb_ids, concept_token_ids = batch_data
                    concept_token_ids = concept_token_ids.to(self.device)
                    concept_embeddings = concept_encoder(concept_token_ids)
                    for kb_id, concept_embedding in zip(kb_ids, concept_embeddings):
                        f_out.write(json.dumps({'id': kb_id, 'vec': concept_embedding.tolist()}) + '\n')