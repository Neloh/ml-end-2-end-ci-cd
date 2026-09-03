"""
Build + run the end-to-end SageMaker pipeline. All account-specific values come from
ENVIRONMENT VARIABLES (set as GitHub repo variables) so NOTHING sensitive is committed.

Required env vars (set as GitHub Actions repo *variables*, not secrets — none are secret):
  AWS_REGION            e.g. us-east-1
  SM_EXEC_ROLE_ARN      SageMaker execution role ARN (passed to training/processing)
  SM_BUCKET             S3 bucket for artifacts (name only)
  MODEL_PACKAGE_GROUP   model registry group name
  ACC_THRESHOLD         optional float, default 0.80
"""
import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.model_step import ModelStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model

REGION   = os.environ["AWS_REGION"]
ROLE     = os.environ["SM_EXEC_ROLE_ARN"]
BUCKET   = os.environ["SM_BUCKET"]
MPG      = os.environ.get("MODEL_PACKAGE_GROUP", "ml-e2e-mpg")
THRESH   = float(os.environ.get("ACC_THRESHOLD", "0.80"))
PREFIX   = "ml-e2e-ci-cd"

pipe_sess = PipelineSession(default_bucket=BUCKET)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")


def build():
    input_data = ParameterString("InputData", default_value=f"s3://{BUCKET}/{PREFIX}/raw/")
    threshold = ParameterFloat("AccThreshold", default_value=THRESH)

    sk = SKLearnProcessor(framework_version="1.2-1", role=ROLE,
                          instance_type="ml.m5.large", instance_count=1, sagemaker_session=pipe_sess)
    prep = ProcessingStep(
        name="DataPrep", processor=sk,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(SRC, "preprocessing.py"),
    )

    img = sagemaker.image_uris.retrieve("xgboost", REGION, "1.7-1")
    est = Estimator(image_uri=img, role=ROLE, instance_type="ml.m5.large", instance_count=1,
                    output_path=f"s3://{BUCKET}/{PREFIX}/models/", sagemaker_session=pipe_sess)
    est.set_hyperparameters(objective="binary:logistic", num_round=50, max_depth=4)
    train = TrainingStep(
        name="Train", estimator=est,
        inputs={
            "train": TrainingInput(prep.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="text/csv"),
            "validation": TrainingInput(prep.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri, content_type="text/csv"),
        },
    )

    eval_report = PropertyFile(name="EvalReport", output_name="evaluation", path="evaluation.json")
    xgb_eval = ScriptProcessor(image_uri=img, command=["python3"], role=ROLE,
                               instance_type="ml.m5.large", instance_count=1, sagemaker_session=pipe_sess)
    evaluate = ProcessingStep(
        name="Evaluate", processor=xgb_eval,
        inputs=[
            ProcessingInput(source=train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=prep.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code=os.path.join(SRC, "evaluate.py"),
        property_files=[eval_report],
    )

    model = Model(image_uri=img, model_data=train.properties.ModelArtifacts.S3ModelArtifacts,
                  role=ROLE, sagemaker_session=pipe_sess)
    register = ModelStep(
        name="Register",
        step_args=model.register(
            content_types=["text/csv"], response_types=["text/csv"],
            inference_instances=["ml.m5.large"], transform_instances=["ml.m5.large"],
            model_package_group_name=MPG, approval_status="PendingManualApproval"),
    )

    gate = ConditionStep(
        name="AccuracyGate",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=evaluate.name, property_file=eval_report,
                         json_path="binary_classification_metrics.accuracy.value"),
            right=threshold)],
        if_steps=[register], else_steps=[])

    return Pipeline(name="ml-e2e-ci-cd-pipeline",
                    parameters=[input_data, threshold],
                    steps=[prep, train, evaluate, gate],
                    sagemaker_session=pipe_sess)


if __name__ == "__main__":
    pipe = build()
    pipe.upsert(role_arn=ROLE)
    ex = pipe.start()
    print(f"Started execution: {ex.arn}")
