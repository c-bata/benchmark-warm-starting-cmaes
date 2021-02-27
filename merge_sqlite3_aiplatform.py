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
        ws_sqlite_filename = f"shibata_wscmaes_db-ws-seed{seed}.sqlite3"

        # source task
        study = optuna.create_study(
            study_name=f"ws-source-seed{seed}",
            storage="sqlite:///wscmaes-toxic-experiments.sqlite3"
        )
        ws_source_study = optuna.load_study(
            study_name=f"source-task-seed{seed}",
            storage=f"sqlite:///db/{ws_sqlite_filename}"
        )
        study.add_trials(ws_source_study.trials)

        # target task
        study = optuna.create_study(
            study_name=f"ws-seed{seed}",
            storage="sqlite:///wscmaes-toxic-experiments.sqlite3"
        )
        ws_target_study = optuna.load_study(
            study_name=f"target-task-seed{seed}",
            storage=f"sqlite:///db/{ws_sqlite_filename}"
        )
        study.add_trials(ws_target_study.trials)


if __name__ == '__main__':
    main()
