# Chest-Cancer-Classification-using-MLFlow-and-DVC

## Workflows

1. Update config.yaml 
2. Update params.yaml
3. Update the entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the dvc.yaml


## ML Flow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### Dagshub

[Dagshub](https://dagshub.com/)


MLFLOW_TRACKING_URI=https://dagshub.com/VedanshArora1905/Chest-Cancer-Classification-using-MLFlow-and-DVC.mlflow \
MLFLOW_TRACKING_USERNAME=VedanshArora1905 \
MLFLOW_TRACKING_PASSWORD=Vedansh@1122 \
python script.py

Run this to export as ENV variables

 ```bash

  
```

### DVC cmd
1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

##### MLflow

Its Production Grade
Trace all of your expriements
Logging & taging your model

##### DVC

Its very lite weight for POC only
lite weight expriements tracker
It can perform Orchestration (Creating Pipelines)

## AWS-CICD-Deployment-with-Github-Actions

1. Login to AWS console.
2. Create IAM user for deployment

``` bash

#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch your EC2 

4. Pull Your image from ECR in EC2

5. Launch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess

```
3. Create ECR repo to store/save docker image

```bash
- Save the URI: 314146305559.dkr.ecr.eu-north-1.amazonaws.com/chest
```

4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install docker in EC2 Machine:

```bash     
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

6. Configure EC2 as self-hosted runner:
```bash
setting>actions>runner>new self hosted runner > choose os > then run command one by one
```
7. Setup github secrets:
```bash
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = 

AWS_ECR_LOGIN_URI = 

ECR_REPOSITORY_NAME = 
```