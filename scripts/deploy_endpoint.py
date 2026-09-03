"""Deploy a real-time endpoint from an approved ModelPackage ARN.
Region + execution role come from env (GitHub repo variables). No hardcoded account values.
"""
import argparse, os, time, boto3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-package-arn", required=True)
    args = ap.parse_args()

    region = os.environ["AWS_REGION"]
    role = os.environ["SM_EXEC_ROLE_ARN"]
    sm = boto3.client("sagemaker", region_name=region)

    ts = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"ml-e2e-model-{ts}"
    ec_name = f"ml-e2e-ec-{ts}"
    ep_name = "ml-e2e-ci-cd-endpoint"

    sm.create_model(ModelName=model_name, ExecutionRoleArn=role,
                    Containers=[{"ModelPackageName": args.model_package_arn}])
    sm.create_endpoint_config(EndpointConfigName=ec_name, ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": model_name,
        "InitialInstanceCount": 1, "InstanceType": "ml.m5.large"}])

    # create if missing, else blue/green update
    existing = [e["EndpointName"] for e in sm.list_endpoints(NameContains=ep_name)["Endpoints"]]
    if ep_name in existing:
        sm.update_endpoint(EndpointName=ep_name, EndpointConfigName=ec_name)
        print(f"UpdateEndpoint {ep_name} -> {ec_name}")
    else:
        sm.create_endpoint(EndpointName=ep_name, EndpointConfigName=ec_name)
        print(f"CreateEndpoint {ep_name} -> {ec_name}")

if __name__ == "__main__":
    main()
