import logging
import os
from typing import Dict, Any, Tuple
from typing import List
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class StringIdTensorDataset(Dataset):
    """
    Holds tensors with a String as identifier, e.g. a CUI.
    """

    def __init__(self, data: List[Tuple[str, torch.Tensor]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_device() -> Tuple[str, int]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = None
    if str(device) == 'cuda':
        torch.cuda.empty_cache()
        n_gpu = torch.cuda.device_count()
        logger.info("Number of GPUs: {}".format(n_gpu))
    return str(device), n_gpu


def get_config_from_env() -> Dict[str, Any]:
    config = {
        'hpo': os.getenv('HPO', 'True').lower() in ['true', '1'],
        'evaluate': os.getenv('EVALUATE', 'True').lower() in ['true', '1'],
        'create_concept_emb': os.getenv('CREATE_CONCEPT_EMB', 'True').lower() in ['true', '1'],
        'ner_batch_size': int(os.getenv('NER_BATCH_SIZE', '16')),
        'bert_model': os.getenv('BERT_MODEL', 'bert-base-german-dbmdz-uncased'),
        'batch_size': int(os.getenv('BATCH_SIZE', '64')),
        'epochs': int(os.getenv('EPOCHS', '100')),
        'learning_rate': float(os.getenv('LEARNING_RATE', '1e-4')),
        'freeze_emb': os.getenv('FREEZE_EMB', 'True').lower() in ['true', '1'],
        'warmup_steps': int(os.getenv('WARMUP_STEPS', '100')),
        'input_length': int(os.getenv('INPUT_LENGTH', '50')),
        'valtest_interval': int(os.getenv('VALTEST_INTERVAL', '1')),
        'force_cpu': os.getenv('FORCE_CPU', 'False').lower() in ['true', '1'],
        'omp_num_threads': int(os.getenv('OMP_NUM_THREADS', 1)),
        'comment': os.getenv('COMMENT', None),
        'fix_random_seed': not os.getenv('FIX_RANDOM_SEED', 'True').lower() in ['false', '0'],
        'clip_grad_norm': os.getenv('CLIP_GRAD_NORM', '').lower() in ['true', '1'],
        'k_nn': int(os.getenv('K_NN', 10)),
        'train_ner': os.getenv('TRAIN_NER', '').lower() in ['true', '1'],
        'model_name': os.getenv('MODEL_NAME', ''),
        'paths': {
            'kb': os.getenv('PATH_KB', '/data/datasets/krohne_products_description_texoo.json'),
            'train': os.getenv('PATH_TRAIN', '/data/datasets/krohne_servicetickets_train_annotations_texoo.json'),
            'test': os.getenv('PATH_TEST', '/data/datasets/krohne_servicetickets_test_annotations_texoo.json'),
            'dev': os.getenv('PATH_DEV', '/data/datasets/krohne_servicetickets_dev_annotations_texoo.json'),
            'model': os.getenv('PATH_MODEL', '/data/biencoder/model/'),
            'tensorboard': os.getenv('PATH_TENSORBOARD', '/data/biencoder/runs/'),
            'ray': os.getenv('PATH_RAY', '/data/biencoder/runs/ray/')
        }
    }

    if config['force_cpu']:
        config['device'] = 'cpu'
        logger.warning('Forcing CPU usage, did not try to acquire GPU!')
    else:
        config['device'], config['num_gpu'] = get_device()
    logger.info('Using {}'.format(config['device']))

    return config



