# ml-end-2-end-ci-cd

End-to-end SageMaker MLOps pipeline driven by GitHub Actions via **OIDC** (no long-lived AWS keys).

Flow: DataPrep -> Train (XGBoost) -> Evaluate -> AccuracyGate (ConditionStep) -> RegisterModel
(PendingManualApproval) -> [manual approval] -> deploy endpoint.

## Security model

- **No AWS keys anywhere.** GitHub Actions federates to AWS with OIDC (`aws-actions/configure-aws-credentials`)
  assuming a role whose trust is pinned to THIS repo (`repo:Neloh/ml-end-2-end-ci-cd:...`).
- **No sensitive values committed.** Account ID, role ARNs, and bucket name are provided at runtime
  through GitHub **repository variables** (below), not in the code.
- Least-privilege IAM roles; `iam:PassRole` locked to a single execution role + `sagemaker.amazonaws.com`.

## Required GitHub repository *Variables* (Settings -> Secrets and variables -> Actions -> Variables)

These are NON-secret config values (safe as plain variables):

| Variable | Value |
|---|---|
| `AWS_REGION` | your region, e.g. `us-east-1` |
| `AWS_PIPELINE_ROLE_ARN` | ARN of the `gha-sagemaker-pipeline` role |
| `AWS_DEPLOY_ROLE_ARN` | ARN of the `gha-sagemaker-deploy` role |
| `SM_EXEC_ROLE_ARN` | your SageMaker execution role ARN |
| `SM_BUCKET` | S3 bucket name for artifacts |
| `MODEL_PACKAGE_GROUP` | e.g. `ml-e2e-mpg` |
| `ACC_THRESHOLD` | e.g. `0.80` |

(The exact ARN/bucket values were provisioned separately and are given to you out-of-band — do not
commit them. They are not secret, but keeping them out of git keeps the repo portable/clean.)

## GitHub setup (one-time)

1. Create the repo `Neloh/ml-end-2-end-ci-cd` and push this folder.
2. Add the repository variables above.
3. Create a GitHub **Environment** named `production` with a required reviewer (this is the manual
   approval gate the deploy workflow uses).
4. Push to `main` -> `build-and-run-pipeline` runs: tests -> OIDC auth -> upsert + start the pipeline.
5. When the model package is approved, run the `deploy` workflow (manual `workflow_dispatch`) with the
   approved ModelPackage ARN; the `production` environment gate requires an approver before it deploys.

## Layout

```
.github/workflows/  build-and-run-pipeline.yml, deploy.yml
pipelines/          definition.py           # builds the SageMaker Pipeline
src/                preprocessing.py, evaluate.py
scripts/            make_raw.py, deploy_endpoint.py
requirements.txt
```

The SageMaker-side of this (the pipeline, Feature Store, model registry) was proven live separately;
this repo adds the GitHub Actions CI/CD outer loop that triggers it via OIDC.
