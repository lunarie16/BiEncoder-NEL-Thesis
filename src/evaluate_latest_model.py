from evaluation.generate_kb_concept_ebmeddings import ConceptEmbGenerator
from evaluation.ner_nel_combined import NerNelEvaluation
import os
import logging
from tensorboardX import SummaryWriter
from evaluation.utils import CalculateMetrics
from ray.tune import report

logger = logging.getLogger(__name__)


def start_evaluating(writer: SummaryWriter,
                     create_concept_emb: bool,
                     model_descriptor: str,
                     model_path: str,
                     trained_concept_encoder_path: str,
                     kb_path: str,
                     input_length: int,
                     batch_size: int,
                     device: str,
                     freeze_embeddings: bool,
                     test_path: str,
                     ner_model_path: str,
                     bert_model: str
                     ):
    concept_emd_path = model_path + 'concept_vectors-' + model_descriptor + '.jsonlines'
    logger.warning('saving concept embeddings as {}'.format(concept_emd_path))
    if create_concept_emb:
        logger.error('\n\n\nStart generating Concept Embeddings Vector File\n')
        generator = ConceptEmbGenerator(model_descriptor=model_descriptor,
                                        model_path=model_path,
                                        trained_concept_encoder_path=trained_concept_encoder_path,
                                        kb_path=kb_path,
                                        jsonlines_out_path=concept_emd_path,
                                        input_length=input_length,
                                        batch_size=batch_size,
                                        device=device,
                                        bert_model=bert_model,
                                        freeze_embeddings=freeze_embeddings,
                                        )
        generator.generate()
        logger.info(f'\n\n\nFile {concept_emd_path} got successfully generated.\n')

    logger.info('\n\n\nStart evaluating NER and NEL Pipeline\n')
    evaluation_exact = NerNelEvaluation(test_path=test_path,
                                        ner_model_path=ner_model_path,
                                        encoder_path=os.path.join(model_path, model_descriptor),
                                        kb_path=kb_path,
                                        encoder_input_length=input_length,
                                        precomputed_concept_embeddings_path=concept_emd_path,
                                        # TODO: hier auch noch anderen FAISSIndex benutzen
                                        index_type=NerNelEvaluation.IndexType.FAISSExactNormalized,
                                        bert_model=bert_model,
                                        device=device,
                                        ignore_cache=True)

    tp, fp, fn = evaluation_exact.run_micro()
    logger.info('\n\n--- resulting micro-average ---')
    logger.info(f'tp: {tp} fp: {fp} fn: {fn}')
    calculator = CalculateMetrics(tp=tp, fp=fp, fn=fn)
    precision, recall, f1_score = calculator.get_all()
    logger.info(f'precision: {precision}, recall: {recall}, f1 score: {f1_score}')

    macro = evaluation_exact.run_macro()
    logger.info(f'\n\n--- resulting macro-average ---')
    for t in macro:
        ty = macro[t]
        tp = ty['tp']
        fp = ty['fp']
        fn = ty['fn']
        logger.info(f'\n\n--- Exact KNN with Macro for {t} ---')
        logger.info(f'Category: {t} tp: {tp} fp: {fp} fn: {fn}')
        calculator = CalculateMetrics(tp=tp, fp=fp, fn=fn)
        precision, recall, f1_score = calculator.get_all()
        logger.info(f'Category: {t} precision: {precision}, recall: {recall}, f1 score: {f1_score}')



