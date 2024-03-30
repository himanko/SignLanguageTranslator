from SignLanguageTranslatorAPP import logger
from SignLanguageTranslatorAPP.pipeline.pradiction_pipeline import PradictionPipeline


STAGE_NAME = "Single stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PradictionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e