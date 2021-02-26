# ref: https://www.kaggle.com/peterhurford/lightgbm-with-select-k-best-on-tfidf
import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import requests

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description="Run a benchmark of Warm Starting CMA-ES.")
parser.add_argument("--seed", type=int, default=1, help="Number of repeat.")
parser.add_argument("--slack-url", type=str, default="", help="Slack Webhook URL")
parser.add_argument("--slack-channel", type=str, default="", help="Slack channel")
parser.add_argument("--job-id", type=str, default="unknown", help="Job ID")
parser.add_argument("--gcs-path", type=str, default="", help="Path of GCS")
parser.add_argument('--no-warm-start', default=False, action='store_true')
args = parser.parse_args()

seed = args.seed
slack_url = args.slack_url
slack_channel = args.slack_channel
gcs_path = args.gcs_path
no_warm_start = args.no_warm_start
job_id = args.job_id

rounds_lookup = {
    'toxic': 140,
    'severe_toxic': 50,
    'obscene': 80,
    'threat': 80,
    'insult': 70,
    'identity_hate': 80
}


def notify_slack(msg):
    if slack_url is None or slack_channel is None:
        print(msg)
        return

    requests.post(
        slack_url,
        data=json.dumps(
            {
                "channel": slack_channel,
                "text": msg,
                "username": "WS-CMA-ES Report",
                "link_names": 1,
            }
        ),
    )


def get_word_vec(train_text):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        max_features=50000)
    word_vectorizer.fit(train_text)
    train_word_features = word_vectorizer.transform(train_text)

    return train_word_features


def get_char_vec(train_text):
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(2, 6),
        max_features=50000)
    char_vectorizer.fit(train_text)
    train_char_features = char_vectorizer.transform(train_text)
    return train_char_features


def load_toxic(raw_data, data_frac):
    train = raw_data.head(int(len(raw_data) * data_frac))
    train_text = train['comment_text']
    train_word_features = get_word_vec(train_text)
    train_char_features = get_char_vec(train_text)

    train_features = hstack([train_char_features, train_word_features])
    train_features.tocsr()
    train.drop('comment_text', axis=1, inplace=True)
    return train_features, train


def objective(trial, train_dataset):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_features = train_dataset[0]
    train_labels = train_dataset[1]
    scores = {}

    for class_name in class_names:
        print(class_name)
        train_target = train_labels[class_name]
        model = LogisticRegression(solver='sag')
        sfm = SelectFromModel(model, threshold=0.2)

        print(train_features.shape)
        train_sparse_matrix = sfm.fit_transform(train_features, train_target)
        print(train_sparse_matrix.shape)
        d_train = lgb.Dataset(train_sparse_matrix, label=train_target)

        params = {
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 8, 128, log=True),
            'bagging_fraction': trial.suggest_float("bagging_fraction", 0.1, 0.9),
            'feature_fraction': trial.suggest_float("feature_fraction", 0.1, 0.9),
            'lambda_l1': trial.suggest_float("lambda_l1", 1e-1, 10, log=True),
            'lambda_l2': trial.suggest_float("lambda_l2", 1e-1, 10, log=True),
            'application': 'binary',
            'verbosity': -1,
            'metric': 'auc',
            'data_random_seed': 2,
            'nthread': 16,
            'seed': seed
        }
        score = lgb.cv(
            params,
            train_set=d_train,
            nfold=5,
            stratified=True,
            num_boost_round=rounds_lookup[class_name],
            verbose_eval=10
        )
        cv_auc = max(score["auc-mean"])
        scores[class_name] = 1. - cv_auc
    return np.mean(list(scores.values()))


def run_wscmaes():
    raw_data = pd.read_csv('tmp/train.csv').fillna(' ')
    sqlite_filename = f"db-ws-seed{seed}.sqlite3"
    storage = f"sqlite:///{sqlite_filename}"

    source_task_train_dataset = load_toxic(raw_data, 0.1)
    target_task_train_dataset = load_toxic(raw_data, 1.0)

    try:
        # Source task
        source_study = optuna.create_study(
            storage=storage,
            study_name=f"source-task-seed{seed}",
            sampler=optuna.samplers.RandomSampler(seed=seed),
        )
        source_study.optimize(lambda t: objective(t, source_task_train_dataset), 100)
        notify_slack(f"Source task finished: seed={seed}\n")

        # 中間結果を保存。
        cmd = [
            "gsutil",
            "cp",
            os.path.join(".", sqlite_filename),
            os.path.join(gcs_path, "source-" + sqlite_filename),
        ]
        subprocess.run(cmd, check=False, timeout=10 * 60)

        # Target task
        source_trials = source_study.trials
        sampler = optuna.samplers.CmaEsSampler(seed=seed, source_trials=source_trials)
        study = optuna.create_study(
            storage=storage,
            study_name=f"target-task-seed{seed}",
            sampler=sampler
        )
        study.optimize(lambda t: objective(t, target_task_train_dataset), 30)
        print(study.best_trial)
        notify_slack(f"Target task finished: seed={seed}\n")

        # Upload to GCS
        cmd = [
            "gsutil",
            "cp",
            os.path.join(".", sqlite_filename),
            os.path.join(gcs_path, sqlite_filename),
        ]
        subprocess.run(cmd, check=True, timeout=10 * 60, )
    except Exception as e:
        notify_slack(f"Job catch an exception: seed={seed} error={e}\n")

    notify_slack(f"Job finished: seed={seed}\n")


def run_normal_cmaes():
    raw_data = pd.read_csv('tmp/train.csv').fillna(' ')
    sqlite_filename = f"db-normal-seed{seed}.sqlite3"
    storage = f"sqlite:///{sqlite_filename}"

    target_task_train_dataset = load_toxic(raw_data, 1.0)
    try:
        # Target task
        study = optuna.create_study(
            storage=storage,
            study_name=f"target-task-seed{seed}",
            sampler=optuna.samplers.CmaEsSampler(seed=seed)
        )
        study.optimize(lambda t: objective(t, target_task_train_dataset), 30)
        print(study.best_trial)
        notify_slack(f"Target task finished: seed={seed}\n")

        # Upload to GCS
        cmd = [
            "gsutil",
            "cp",
            os.path.join(".", sqlite_filename),
            os.path.join(gcs_path, sqlite_filename),
        ]
        subprocess.run(cmd, check=True, timeout=10 * 60)
    except Exception as e:
        notify_slack(f"Job catch an exception: seed={seed} error={e}\n")

    notify_slack(f"Job finished: seed={seed}\n")


if __name__ == '__main__':
    if no_warm_start:
        run_normal_cmaes()
    else:
        run_wscmaes()
