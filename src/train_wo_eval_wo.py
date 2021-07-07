import logging
import os

import numpy
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from biencodernel.biencoder import BiEncoder
from biencodernel.datasets import KBDataset, MentionDatasetNoNER
from biencodernel.utils import get_config_from_env

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    config = get_config_from_env()

    if config['fix_random_seed']:
        torch.manual_seed(0)
        numpy.random.seed(0)

    kb_ds = KBDataset(path_to_json=config['paths']['kb'], max_length=20)
    train_ds = MentionDatasetNoNER(path_to_json=config['paths']['train'], max_length=20, kb_dataset=kb_ds)
    test_ds = MentionDatasetNoNER(path_to_json=config['paths']['test'], max_length=20, kb_dataset=kb_ds)
    dev_ds = MentionDatasetNoNER(path_to_json=config['paths']['dev'], max_length=20, kb_dataset=kb_ds)

    kb_dl = DataLoader(dataset=kb_ds, batch_size=config['batch_size'], drop_last=True, pin_memory=True)
    train_dl = DataLoader(dataset=train_ds, sampler=RandomSampler(train_ds), batch_size=config['batch_size'], drop_last=True, pin_memory=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=config['batch_size'], drop_last=True, pin_memory=True)
    dev_dl = DataLoader(dataset=dev_ds, batch_size=config['batch_size'], drop_last=True, pin_memory=True)

    model_descriptor = 'train_wo_eval_wo_il{}-bs{}-lr{}-wu{}-ep{}{}-{}'.format(
        config['input_length'],
        config['batch_size'],
        config['learning_rate'],
        config['warmup_steps'],
        config['epochs'],
        ('-' + config['comment']) if config['comment'] is not None else '',
        str(config['device']),
    )
    logger.info('Model descriptor: {}'.format(model_descriptor))
    os.makedirs(config['paths']['tensorboard'], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config['paths']['tensorboard'], model_descriptor))

    biencoder = BiEncoder(
        device=config['device'],
        model_path=config['paths']['model'],
        writer=writer,
        tokenizer=train_ds.tokenizer,
        freeze_embeddings=config['freeze_emb']
    )

    biencoder.train(
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        dev_dataloader=dev_dl,
        kb_dataloader=kb_dl,
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        valtest_interval=config['valtest_interval'],
        epochs=config['epochs'],
        omp_num_threads=config['omp_num_threads'],
        clip_grad_norm=config['clip_grad_norm']
    )

    biencoder.save_encoders(config['paths']['model'], model_descriptor)