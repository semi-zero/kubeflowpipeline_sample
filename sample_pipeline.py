import kfp
from kfp.components import InputPath, OutputPath
import requests
import kfp.dsl as dsl
import datetime
from typing import NamedTuple

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

import yaml

def read_data(csv_url: str,
              target: str,
              unique_id: str,
              minio_info: dict,
              learn_minio_path: str,
              output_df: OutputPath(),
              var_json: OutputPath(), ) -> None:
    import pandas as pd
    import json
    import datetime
    import psycopg2
    from minio import Minio
    from io import BytesIO
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import namedtuple

 
    # minio 정보
    minio_endpoint = minio_info["minio_endpoint"]
    minio_access = minio_info["minio_access"]
    minio_secret = minio_info["minio_secret"]
    bucket_name = minio_info["bucket_name"]

    # 파일 이름
    file_name = csv_url.split(".")

    # df = pd.read_csv(csv_url)
    minio_client = Minio(minio_endpoint, access_key=minio_access, secret_key=minio_secret, secure=False,
                        region="us-west-rack2")

    import re
        
    response = minio_client.get_object(bucket_name=learn_minio_path, object_name=csv_url)
    csv_file = BytesIO(response.data)
    csv_file_backup = BytesIO(response.data)
    csv_length = len(csv_file.getvalue())
    minio_client.put_object(bucket_name=learn_minio_path,
                            object_name=csv_url,
                            data=csv_file_backup, 
                            length=csv_length, 
                            content_type="application/csv")
    print(response)
    print(csv_file)
    response.close()
    response.release_conn()
    
   
    
    args_list = [{},
    {"encoding": "cp949"},
    {"encoding": "utf-8"},
    {"encoding": "utf-16"},
    {"encoding": "euc-kr"}]
    
    check = False
    for args in args_list:
        try:
            df = pd.read_csv(csv_file, **args)
            check = True
        except:
            continue


    df = df.sort_index(axis=1)
    var_list = df.columns.tolist()
    num_var = sorted(list(set(df.select_dtypes(include='float').columns) | set(df.select_dtypes(include='int').columns)))
    num_var = sorted([x for x in num_var if x not in [target] + [unique_id]])
    obj_var = sorted([x for x in var_list if x not in num_var + [target] + [unique_id]])
    
    var = {"var_list": var_list,
        "num_var": num_var,
        "obj_var": obj_var}


    with open(var_json, 'w') as f:
        json.dump(var, f)

    with open(output_df, "w") as f:
        df.to_csv(f, index=False)


def preprocess_data(data: InputPath(),
                    var_json: InputPath(),
                    target: str,
                    unique_id: str,
                    preprocess_output_df: OutputPath(),
                    ) -> None:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import collections
    import json
    from collections import namedtuple

 
    with open(data, 'rb') as f:
        df = pd.read_csv(f)

    with open(var_json, 'r') as f:
        var_json = json.load(f)

    var_list = var_json['var_list']
    num_var = var_json['num_var']
    obj_var = var_json['obj_var']

    print(var_list)

    # 1. id_target_preprocess
    # target label encoding
    if target in df.select_dtypes(exclude=['float', 'int']).columns.tolist():
        lb_encoder = LabelEncoder()
        df.loc[:, target] = lb_encoder.fit_transform(df.loc[:, target])

    id_target_df = df[[unique_id] + [target]]
    df = df.drop([unique_id] + [target], axis=1)

    # 2. na_preprocess
    # 문자형 변수에 결측치가 있다면 "NaN"으로 보간
    if obj_var:
        obj_data = df.loc[:, obj_var]
        if np.sum(obj_data.isnull().sum()) != 0:
            obj_data.fillna("NaN", inplace=True)

    # 수치형 변수에 결측치가 있다면 "평균치"로 보간
    if num_var:
        num_data = df.loc[:, num_var]
        if np.sum(num_data.isnull().sum()) != 0:
            num_data.fillna(num_data.mean(), inplace=True)

    df = pd.concat([obj_data, num_data], axis=1)

    # 3. label_encoder
    if obj_var:
        obj_data = df.loc[:, obj_var]
        non_obj_data = df.drop(set(obj_var), axis=1)

        lbl_en = LabelEncoder()
        lbl_en = collections.defaultdict(LabelEncoder)
        obj_data = obj_data.apply(lambda x: lbl_en[x.name].fit_transform(x))

        # pickle.dump(lbl_en, open('storage/model/label_encoder.pkl', 'wb')) => 라벨 인코더 저장해야된다....!!
        df = pd.concat([obj_data, non_obj_data], axis=1)

    # 4. standardize
    if num_var:
        num_data = df.loc[:, num_var]
        non_num_data = df.drop(set(num_var), axis=1)

        std_scaler = StandardScaler()
        fitted = std_scaler.fit(num_data)
        output = std_scaler.transform(num_data)
        num_data = pd.DataFrame(output, columns=num_data.columns, index=list(num_data.index.values))

        df = pd.concat([non_num_data, num_data], axis=1)

    # 5. get_df
    df = pd.concat([df, id_target_df], axis=1)

    with open(preprocess_output_df, "w") as f:
        df.to_csv(f, index=False)



def prepare_modeling(data: InputPath(),
                     target: str,
                     unique_id: str,
                     train_test_json: OutputPath(),
                     evaluations_dict_json: OutputPath(),
                     ) -> None:
    import pandas as pd
    import json
    from sklearn.model_selection import train_test_split
    from collections import namedtuple

    with open(data, 'rb') as f:
        df = pd.read_csv(f)

    id_df = df[[unique_id]]
    df = df.drop([unique_id], axis=1)

    df_y = pd.DataFrame(df.loc[:, target])
    df_x = df.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=20)

    y_train.reset_index(inplace=True)
    y_test.reset_index(inplace=True)

    train_test = {'df': df.to_json(),
                'X_train': X_train.to_json(),
                'X_test': X_test.to_json(),
                'y_train': y_train.to_json(),
                'y_test': y_test.to_json()}

    with open(train_test_json, 'w') as f:
        json.dump(train_test, f)

    evaluations_dict = dict()
    with open(evaluations_dict_json, 'w') as f:
        json.dump(evaluations_dict, f)


def rf_modeling(train_test: InputPath(),
                rf_model_pkl: OutputPath(),
                ) -> None :
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import StratifiedKFold, KFold
    import numpy as np
    import pandas as pd
    import json
    import pickle
    import optuna
    from optuna import Trial
    from optuna.samplers import TPESampler
    from collections import namedtuple


    with open(train_test, 'r') as f:
        train_test = json.load(f)

    X_train = pd.read_json(train_test['X_train'])
    y_train = pd.read_json(train_test['y_train'])
    y_train = y_train.set_index('index')

    
    parameters = {'n_estimators': 10, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}#, 'n_jobs': 4}

    rf = RandomForestClassifier(**parameters)
    rf_reg = rf.fit(X_train, y_train)

    with open(rf_model_pkl, 'wb') as f:
        pickle.dump(rf_reg, f)


def evaluate_model(model_type: str,
                   train_test: InputPath(),
                   model_pth: InputPath(),
                   evaluations_dict_json: InputPath(),
                   out_evaluations_dict_json: OutputPath(),
                    )-> None:
    import pandas as pd
    import numpy as np
    import json
    import math
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
        average_precision_score, confusion_matrix

    import pickle
    from collections import namedtuple

    with open(train_test, 'r') as f:
        train_test = json.load(f)

    X_test = pd.read_json(train_test['X_test'])
    y_test = pd.read_json(train_test['y_test'])
    y_test = y_test.set_index('index')

    X_test_values = X_test.values
    y_test_values = y_test.values.reshape(-1, )

    # 모델 별 예측
    with open(model_pth, 'rb') as f:
        model = pickle.load(f)
    

    y_pred = model.predict(X_test_values)
    y_pred_proba = model.predict_proba(X_test_values)[:, 1]

    # model_type별 score
    with open(evaluations_dict_json, 'r') as f:
        evaluations_dict = json.load(f)

    evaluation_dict = dict()
    evaluation_dict['model_type'] = model_type
    evaluation_dict['AUROC'] = np.round(roc_auc_score(y_test, y_pred_proba), 3)
    evaluation_dict['AUCPR'] = np.round(average_precision_score(y_test, y_pred_proba), 3)
    evaluation_dict["Accuracy"] = np.round(accuracy_score(y_test, y_pred), 3)
    evaluation_dict["Recall"] = np.round(recall_score(y_test, y_pred), 3)
    evaluation_dict["Precision"] = np.round(precision_score(y_test, y_pred), 3)
    evaluation_dict["F1"] = np.round(f1_score(y_test, y_pred), 3)
  
    # model_type별 score 저장
    evaluations_dict[model_type] = evaluation_dict


    with open(out_evaluations_dict_json, 'w') as f:
        json.dump(evaluations_dict, f)





#local 이미지
BASE_IMAGE_URL = 'python:3.9'
base_image_url = BASE_IMAGE_URL


###################create component########################
read_data_comp = kfp.components.create_component_from_func(
    func=read_data,
    base_image=base_image_url,
    packages_to_install=['pandas==1.3.5', 'numpy==1.23.0', 'psycopg2-binary==2.9.7', 'minio==7.1.3', 'pyarrow']
)


preprocess_data_comp = kfp.components.create_component_from_func(
    func=preprocess_data,
    base_image=base_image_url,
    packages_to_install=['pandas==1.3.5', 'numpy==1.23.0', 'scikit-learn==1.0.2']
)

prepare_modeling_comp = kfp.components.create_component_from_func(
    func=prepare_modeling,
    base_image=base_image_url,
    packages_to_install=['pandas==1.3.5', 'numpy==1.23.0', 'scikit-learn==1.0.2', 'imbalanced-learn==0.10.1']
)

rf_modeling_comp = kfp.components.create_component_from_func(
    func=rf_modeling,
    base_image=base_image_url,
    packages_to_install=['pandas==1.3.5', 'numpy==1.23.0', 'scikit-learn==1.0.2', 'optuna==2.10.1']
)


evaluate_model_comp = kfp.components.create_component_from_func(
    func=evaluate_model,
    base_image=base_image_url,
    packages_to_install=['pandas==1.3.5', 'numpy==1.23.0', 'scikit-learn==1.0.2', 'lightgbm']
)




##dsl.pipeline
@dsl.pipeline(
    name='practice',
    description='sample pipeline'
)
def classifier_pipeline(csv_url: str, 
                        target: str, 
                        unique_id: str, 
                        hpo: bool,
                        learn_minio_path: str, 
                        minio_info: dict,
                        ):
    
    read_data_task = read_data_comp(csv_url=csv_url,
                                    minio_info=minio_info,
                                    target=target,
                                    unique_id=unique_id,
                                    learn_minio_path=learn_minio_path)
           
    preprocess_data_task = preprocess_data_comp(data=read_data_task.outputs['output_df'],
                                                var_json=read_data_task.outputs['var_json'],
                                                target=target,
                                                unique_id=unique_id, )
                
           
    prepare_modeling_task = prepare_modeling_comp(data=preprocess_data_task.outputs['preprocess_output_df'],
                                                target=target,
                                                unique_id=unique_id, )
           

    rf_modeling_task = rf_modeling_comp(train_test=prepare_modeling_task.outputs['train_test_json'], )
                    
            

    evaluate_model_task = evaluate_model_comp(model_type='rf',
                                            train_test=prepare_modeling_task.outputs['train_test_json'],
                                            model_pth=rf_modeling_task.outputs['rf_model_pkl'],
                                            evaluations_dict_json=prepare_modeling_task.outputs['evaluations_dict_json'],
                                            )

                    
           
if __name__ == "__main__":
    
    #######################local에서 실험 방식 => minikube가 설치되어 있어야 함#######################
    # client 방식 => 현재 .py로 외부에서 접근하는 것이기 때문에 port-forwarding을 하고 localhost로 연결
    HOST = "http://localhost:8080"
    USERNAME = "user@example.com"
    PASSWORD = "12341234"
    NAMESPACE = "kubeflow-user-example-com"

    session = requests.Session()
    response = session.get(HOST)
    print(response)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    print(response.url)
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]

    client = kfp.Client(
        host=f"{HOST}/pipeline",
        cookies=f"authservice_session={session_cookie}",
        namespace=NAMESPACE,
    )
    
    #인수값 지정
    learn_minio_path = 'mlpipeline'                 # 버킷 경로
    csv_url = 'data/sample.csv'                     # 데이터 경로 
    target = 'churn'                                # 타겟 변수
    unique_id = 'state'                             # 식별자 변수
    model_type = 'rf'                               # 알고리즘 종류
    
    
    #local일 경우 service에 직접 연결
    minio_info = {"minio_endpoint": "minio-service.kubeflow:9000", 
                  "minio_access": "minio", 
                  "minio_secret": "minio123",
                  "bucket_name": "mlpipeline"}


    #pipeline 가동을 위한 인수들
    pipeline_func = classifier_pipeline
    namespace = 'kubeflow-user-example-com'
    experiment_name = NAMESPACE + '-experiment'
    pipeline_name = 'practice-pipeline'
    run_name = pipeline_name + f' One-off'
    arguments = {"csv_url": csv_url,
                 "learn_minio_path": learn_minio_path,
                 "target": target,
                 "unique_id": unique_id,
                 "minio_info": minio_info,
                 }

    run_result = client.create_run_from_pipeline_func(pipeline_func,
                                                      experiment_name=experiment_name,
                                                      namespace=namespace,
                                                      run_name=run_name,
                                                      arguments=arguments)
    
    #######################개발환경 파이프라인 업로드 방식#######################
    #kfp.compiler.Compiler().compile(pipeline_func=classifier_pipeline, package_path='churn_pipeline.yaml')