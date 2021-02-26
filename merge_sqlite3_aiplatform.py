import optuna


def main():
    for seed in range(1, 13):
        study = optuna.create_study(
            study_name=f"normal-seed{seed}",
            storage="sqlite:///wscmaes-toxic-experiments.sqlite3"
        )

        normal_sqlite_filename = f"shibata_wscmaes_db-normal-seed{seed}.sqlite3"
        normal_study = optuna.load_study(
            study_name=f"target-task-seed{seed}",
            storage=f"sqlite:///db/{normal_sqlite_filename}"
        )

        study.add_trials(normal_study.trials)

    for seed in range(1, 13):
        study = optuna.create_study(
            study_name=f"ws-seed{seed}",
            storage="sqlite:///wscmaes-toxic-experiments.sqlite3"
        )

        normal_sqlite_filename = f"shibata_wscmaes_db-ws-seed{seed}.sqlite3"
        normal_study = optuna.load_study(
            study_name=f"target-task-seed{seed}",
            storage=f"sqlite:///db/{normal_sqlite_filename}"
        )

        study.add_trials(normal_study.trials)


if __name__ == '__main__':
    main()
