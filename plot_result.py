import numpy as np
import optuna

from matplotlib import pyplot as plt

n_repeat = 12


def main():
    storage = "sqlite:///wscmaes-toxic-experiments.sqlite3"

    normal_objectives = np.zeros(shape=(n_repeat, 30), dtype=float)
    for i in range(n_repeat):
        seed = i + 1
        normal_study = optuna.load_study(
            study_name=f"normal-seed{seed}",
            storage=storage
        )
        normal_objectives[i] = np.minimum.accumulate(np.array([t.value for t in normal_study.trials], dtype=float))
    mean_normal = list(normal_objectives.mean(axis=0))

    ws_objectives = np.zeros(shape=(n_repeat, 30), dtype=float)
    for i in range(n_repeat):
        seed = i + 1
        ws_study = optuna.load_study(
            study_name=f"ws-seed{seed}",
            storage=storage
        )
        ws_objectives[i] = np.minimum.accumulate(np.array([t.value for t in ws_study.trials], dtype=float))
    mean_ws = list(ws_objectives.mean(axis=0))

    plt.style.use("ggplot")
    _, ax = plt.subplots()
    ax.set_title("HPO of LightGBM on full Toxic Challenge data")
    ax.set_xlabel("Trials")
    ax.set_ylabel("1 - Validation Mean AUC")  # 1-auc
    cmap = plt.get_cmap("tab10")
    x = list(range(1, 31))
    ax.plot(
        x,
        mean_normal,
        marker="o",
        color=cmap(3),
        alpha=0.5,
        label="CMA-ES",
    )
    ax.plot(
        x,
        mean_ws,
        marker="x",
        color=cmap(0),
        alpha=0.5,
        label="WS-CMA-ES",
    )
    ax.legend()
    plt.savefig("./result.png")


if __name__ == '__main__':
    main()
