import logging
import os
import torch
import numpy
import ray
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from biencodernel.biencoder import BiEncoder
from biencodernel.datasets import KBDataset, MentionDataset
from biencodernel.utils import get_config_from_env
from biencodernel.ner import NER, teXoo2CoNLLFile
from evaluate_latest_model import start_evaluating
from hyperopt import hp
from hyperopt.pyll import scope
from flair.hyperparameter.param_selection import SearchSpace, Parameter

def train_ner(search_space: SearchSpace = None):
    os.makedirs(conf['paths']['tensorboard'], exist_ok=True)

    logger.error(msg='Training NER')
    ner = NER(
        device=conf['device'],
        model_base_path=conf['paths']['model'],
        max_epochs=conf['epochs'],
        batch_size=conf['ner_batch_size'],
        search_space=search_space,
        tensorboard_path=conf['paths']['tensorboard']
    )
    ner.train(
        train_texoo=conf['paths']['train'],
        test_texoo=conf['paths']['test'],
        dev_texoo=conf['paths']['dev'],
    )


def train_biencoder(config=None, checkpoint_dir=None):
    if config:
        logger.error(f'found configs with {config}')
        conf['batch_size'] = config['batch_size']
        conf['learning_rate'] = config['learning_rate']
        conf['epochs'] = config['epochs']
        # conf['input_length'] = config['input_length']

    if conf['fix_random_seed']:
        torch.manual_seed(0)
        numpy.random.seed(0)

    kb_ds = KBDataset(path_to_json=conf['paths']['kb'], max_length=conf['input_length'],
                      bert_model=conf['bert_model'])
    train_ds = MentionDataset(path_to_json=conf['paths']['train'], max_length=conf['input_length'],
                              kb_dataset=kb_ds, bert_model=conf['bert_model'])
    test_ds = MentionDataset(path_to_json=conf['paths']['test'], max_length=conf['input_length'], kb_dataset=kb_ds,
                             bert_model=conf['bert_model'])
    dev_ds = MentionDataset(path_to_json=conf['paths']['dev'], max_length=conf['input_length'], kb_dataset=kb_ds,
                            bert_model=conf['bert_model'])

    kb_dl = DataLoader(dataset=kb_ds, batch_size=conf['batch_size'], drop_last=True, pin_memory=True)
    train_dl = DataLoader(dataset=train_ds, sampler=RandomSampler(train_ds), batch_size=conf['batch_size'],
                          drop_last=True, pin_memory=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=conf['batch_size'], drop_last=True, pin_memory=True)
    dev_dl = DataLoader(dataset=dev_ds, batch_size=conf['batch_size'], drop_last=True, pin_memory=True)

    os.makedirs(conf['paths']['tensorboard'], exist_ok=True)
    model_descriptor = get_model_descriptor(conf)
    writer = SummaryWriter(log_dir=os.path.join(conf['paths']['tensorboard'], model_descriptor))

    biencoder = BiEncoder(
        device=conf['device'],
        model_path=conf['paths']['model'],
        writer=writer,
        tokenizer=train_ds.tokenizer,
        freeze_embeddings=conf['freeze_emb'],
        bert_model=conf['bert_model'],
        hpo=conf['hpo']
    )

    recallAt1 = biencoder.train(
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        dev_dataloader=dev_dl,
        kb_dataloader=kb_dl,
        learning_rate=conf['learning_rate'],
        warmup_steps=conf['warmup_steps'],
        valtest_interval=conf['valtest_interval'],
        epochs=conf['epochs'],
        omp_num_threads=conf['omp_num_threads'],
        clip_grad_norm=conf['clip_grad_norm']
    )

    biencoder.save_encoders(conf['paths']['model'], model_descriptor)


def run_hpo_ner():
    search_space = SearchSpace()

    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[2, 4, 6, 8])
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.01, 0.1, 0.15, 0.5])

    logger.error(f'running hpo for NER with space {search_space}')
    train_ner(search_space)


def run_hpo_biencoder():
    defaults = [{
        'batch_size': 64,
        'learning_rate': 1e-4,
        'epochs': 180,
        'input_length': 50,
    }]

    space = {
        'batch_size': scope.int(hp.quniform('batch_size', 36, 128, 2)),
        'learning_rate': hp.loguniform('learning_rate', numpy.log(1e-5), numpy.log(1e-3)),
        'epochs': scope.int(hp.quniform('epochs', 50, 150, 5)),
        # 'input_length': scope.int(hp.quniform('input_length', 20, 100, 5)),
    }

    search = HyperOptSearch(
        space,
        metric="recallAt1",
        mode="max",
        points_to_evaluate=defaults,
    )
    config = {
        "num_samples": 15,
        "resources_per_trial": {"cpu": 1, "gpu": 1},
    }

    scheduler = AsyncHyperBandScheduler(
        metric="recallAt1",
        mode="max",
        grace_period=25,
        max_t=defaults[0]['epochs'],
    )

    analysis = ray.tune.run(
        train_biencoder,
        search_alg=search,
        local_dir=os.path.join(conf['paths']['ray']),
        scheduler=scheduler,
        **config
    )
    logger.info("best config: ", analysis.get_best_config(metric="recallAt1", mode="max"))


def get_model_descriptor(conf):
    model_descriptor = 'train_default_{}{}nbs{}-il{}-bs{}-lr{}-wu{}-ep{}-{}{}-{}'.format(
        ('eval-' if conf['evaluate'] else ''),
        ('ner-' if conf['train_ner'] else ''),
        conf['ner_batch_size'],
        conf['input_length'],
        conf['batch_size'],
        conf['learning_rate'],
        conf['warmup_steps'],
        conf['epochs'],
        conf['bert_model'].split('-')[-1],
        ('-' + conf['comment']) if conf['comment'] is not None else '',
        str(conf['device']),
    )
    logger.error('Model descriptor: {}'.format(model_descriptor))

    return model_descriptor


def evaluate():
    if conf['model_name']:
        model_descriptor = conf['model_name']
    else:
        model_descriptor = get_model_descriptor(conf)
    writer = SummaryWriter(log_dir=os.path.join(conf['paths']['tensorboard'], model_descriptor))
    create_concept_embedding = conf['create_concept_emb']
    start_evaluating(writer=writer,
                     create_concept_emb=create_concept_embedding,
                     model_descriptor=model_descriptor,
                     model_path=conf['paths']['model'],
                     trained_concept_encoder_path=conf['paths']['model'] + model_descriptor +
                                                  '/encoder_concept.statedict',
                     kb_path=conf['paths']['kb'],
                     input_length=conf['input_length'],
                     # TODO: think about nbs or bs
                     batch_size=conf['batch_size'],
                     device=conf['device'],
                     freeze_embeddings=conf['freeze_emb'],
                     test_path=conf['paths']['test'],
                     ner_model_path=os.path.join(conf['paths']['model'], 'ner/best-model.pt'),
                     bert_model=conf['bert_model']
                     )


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    conf = get_config_from_env()

    if conf['hpo']:
        if ("RAY_HEAD_SERVICE_HOST" not in os.environ
                or os.environ["RAY_HEAD_SERVICE_HOST"] == ""):
            raise ValueError("RAY_HEAD_SERVICE_HOST environment variable empty."
                             "Is there a ray cluster running?")
        ray.init(address=os.getenv("RAY_HEAD_SERVICE_HOST") + ":6379")
        logger.error(f'Ray is initialized: {ray.is_initialized()}')
        logger.error(f'running HPO with ray.tune')
        # not working, seems ray tune does not work well with flair
        if conf['train_ner']:
            run_hpo_ner()
        run_hpo_biencoder()
        if conf['evaluate']:
            evaluate()
    else:
        if conf['train_ner']:
            train_ner(None)
        if not conf['model_name']:
            train_biencoder(None)
        if conf['model_name'] and conf['evaluate']:
            evaluate()
        if conf['evaluate']:
            evaluate()

