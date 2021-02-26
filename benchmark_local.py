# ref: https://www.kaggle.com/peterhurford/lightgbm-with-select-k-best-on-tfidf
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

rounds_lookup = {
    'toxic': 140,
    'severe_toxic': 50,
    'obscene': 80,
    'threat': 80,
    'insult': 70,
    'identity_hate': 80
}


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


def objective(trial, seed, train_dataset):
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
            'nthread': 4,
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


def main():
    n_repeat = 12
    storage = "sqlite:///wscmaes-toxic-experiments.sqlite3"
    raw_data = pd.read_csv('tmp/train.csv').fillna(' ')
    source_task_train_dataset = load_toxic(raw_data, 0.1)
    target_task_train_dataset = load_toxic(raw_data, 1.0)

    for seed in range(n_repeat):
        # WS-CMA-ES: Source task
        source_study = optuna.create_study(
            storage=storage,
            study_name=f"ws-source-seed{seed}",
            sampler=optuna.samplers.RandomSampler(seed=seed),
        )
        source_study.optimize(lambda t: objective(t, seed, source_task_train_dataset), 100)

        # WS-CMA-ES: Target task
        source_trials = source_study.trials
        sampler = optuna.samplers.CmaEsSampler(seed=seed, source_trials=source_trials)
        study = optuna.create_study(
            storage=storage,
            study_name=f"ws-seed{seed}",
            sampler=sampler
        )
        study.optimize(lambda t: objective(t, seed, target_task_train_dataset), 30)
        print(study.best_trial)

    for seed in range(n_repeat):
        # CMA-ES
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
        study = optuna.create_study(
            storage=storage,
            study_name=f"normal-seed{seed}",
            sampler=sampler
        )
        study.optimize(lambda t: objective(t, seed, target_task_train_dataset), 30)
        print(study.best_trial)


if __name__ == '__main__':
    main()
