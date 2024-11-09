from ChestCancerClassifier.config.configuration import ConfigurationManager
from ChestCancerClassifier.components.model_eval_mlflow import Evaluation
from ChestCancerClassifier import logger


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def _init_(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()




if _name_ == '_main_':
    try:
        logger.info(f"*")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e