from __future__ import annotations

"""
train.py テンプレート

目的:
- 提供Notebookの "best_xgbmodel 構築パス" をヘッドレスで再現する
- そのうえで、アプリ推論用の安定アーティファクトを保存する

前提 / 仮定:
- DASS CSV は Notebook と同じ列構成
- 外部検証(newCE)も Notebook と同じ構成で渡す
- Notebook全体の探索セルは再現せず、best_xgbmodel を作る本線だけ再現する
- SHAP は Notebookと同様に summary_plot を保存する
"""

import argparse
import copy
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from boruta import BorutaPy
from optuna.samplers import TPESampler
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


DEFAULT_XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 100000,
    "verbosity": 1,
    "booster": "gbtree",
    "n_jobs": 1,
    "gamma": 0,
    "min_child_weight": 1,
    "max_delta_step": 0,
    "subsample": 1,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "reg_alpha": 0.5,
    "reg_lambda": 0.1,
    "scale_pos_weight": 1,
    "base_score": None,
    "random_state": 0,
    "missing": float("nan"),
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "early_stopping_rounds": 15,
}


def convert_to_y(ec3: pd.Series | np.ndarray, mw: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(mw, dtype=float) / np.asarray(ec3, dtype=float))


def convert_to_ec3(y: np.ndarray, mw: np.ndarray) -> np.ndarray:
    return np.asarray(mw, dtype=float) / (10.0 ** np.asarray(y, dtype=float))


def llna_label(ec3: float, num: float = 2.0) -> int:
    if ec3 == 150:
        return 0
    if ec3 <= num:
        return 2
    return 1


def read_project_csv(path: str | Path, encoding: str = "cp932") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)


def split_notebook_style(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Notebook と同じ分割:
    - 最後の列: y (LLNA_EC3)
    - 先頭6列は識別列
    - 7列目以降〜最後の1列手前: X
    """
    name_mw_x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x = name_mw_x.iloc[:, 6:]
    return name_mw_x, x, y


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("inputer", KNNImputer(n_neighbors=5, weights="uniform")),
            ("standarization", StandardScaler()),
        ]
    )


def fit_global_boruta(
    x_full: pd.DataFrame,
    y_data: np.ndarray,
    boruta_perc: int,
) -> np.ndarray:
    pipe = make_pipeline()
    x_pre = pipe.fit_transform(x_full)

    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    boruta = BorutaPy(
        rf,
        n_estimators="auto",
        verbose=0,
        random_state=0,
        perc=boruta_perc,
    )
    boruta.fit(x_pre, y_data)
    return np.asarray(boruta.support_, dtype=bool)


def optuna_tune_xgb(
    x_train_selected: np.ndarray,
    y_train: np.ndarray,
    y_train_label: np.ndarray,
    n_trials: int,
) -> dict[str, Any]:
    best_iterators_list: list[list[int]] = []

    def objective(trial: optuna.Trial) -> float:
        search_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }

        params = {**DEFAULT_XGB_PARAMS, **search_params}
        xgb = XGBRegressor(**params)

        inner_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        cv_scores: list[float] = []
        inner_best_iterations: list[int] = []

        for in_tr_idx, val_idx in inner_skf.split(x_train_selected, y_train_label):
            x_tr = x_train_selected[in_tr_idx]
            x_val = x_train_selected[val_idx]
            y_tr = y_train[in_tr_idx]
            y_val = y_train[val_idx]

            xgb.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=0)

            y_val_pred = xgb.predict(x_val)
            inner_best_iterations.append(int(xgb.best_iteration))
            cv_scores.append(float(r2_score(y_val, y_val_pred)))

        best_iterators_list.append(inner_best_iterations)
        return float(np.mean(cv_scores))

    study = optuna.create_study(
        sampler=TPESampler(seed=0),
        direction="maximize",
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = {**DEFAULT_XGB_PARAMS, **study.best_params}
    best_params["n_estimators"] = max(best_iterators_list[study.best_trial.number])
    del best_params["early_stopping_rounds"]

    return {
        "best_params": best_params,
        "study": study,
        "best_iterators_list": best_iterators_list,
    }


def compute_ad_mask(
    x_new_selected: np.ndarray,
    x_train_selected: np.ndarray,
    knn_k: int,
    ad_coverage: float,
) -> np.ndarray:
    x_dist = cdist(x_train_selected, x_train_selected)
    x_dist_sorted = np.sort(x_dist, axis=1)
    knn_dist = np.mean(x_dist_sorted[:, 1 : knn_k + 1], axis=1)
    knn_dist = np.sort(knn_dist)

    threshold_idx = round(x_train_selected.shape[0] * (1.0 - ad_coverage)) - 1
    threshold_idx = max(0, min(threshold_idx, len(knn_dist) - 1))
    ad_threshold = knn_dist[threshold_idx]

    x_new_dist = cdist(x_new_selected, x_train_selected)
    x_new_dist_sorted = np.sort(x_new_dist, axis=1)
    knn_dist_new = np.mean(x_new_dist_sorted[:, :knn_k], axis=1)
    return knn_dist_new <= ad_threshold


def build_artifact(
    full_feature_columns: list[str],
    selected_mask: np.ndarray,
    folds: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected_feature_columns = [
        c for c, flag in zip(full_feature_columns, selected_mask) if bool(flag)
    ]
    return {
        "artifact_version": 1,
        "source": {
            "created_by": "train.py",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "target": {
            "raw_target_column": "LLNA_EC3",
            "y_definition": "log10(MolWeight / EC3)",
            "ec3_inverse_definition": "MolWeight / (10 ** y)",
        },
        "config": copy.deepcopy(config),
        "xgboost_base_params": copy.deepcopy(DEFAULT_XGB_PARAMS),
        "full_feature_columns": full_feature_columns,
        "selected_feature_mask": selected_mask.tolist(),
        "selected_feature_columns": selected_feature_columns,
        "folds": folds,
        "ad_config": {
            "knn_k": config["AD_KNN_k"],
            "AD_coverage": config["AD_coverage"],
        },
    }


def train_and_export(args: argparse.Namespace) -> Path:
    dass_df = read_project_csv(args.dass_csv, encoding=args.encoding)
    _, dass_x_full, dass_ec3 = split_notebook_style(dass_df)

    if "MolWeight" not in dass_x_full.columns:
        raise ValueError("DASS の説明変数列に MolWeight がありません。")

    exval_x_full = None
    exval_ec3 = None
    if args.exval_csv:
        exval_df = read_project_csv(args.exval_csv, encoding=args.encoding)
        exval_df = exval_df.dropna(subset=["MolWeight"], axis=0).reset_index(drop=True)
        _, exval_x_full, exval_ec3 = split_notebook_style(exval_df)

    y_data = convert_to_y(dass_ec3, dass_x_full["MolWeight"].values)
    y_reconverted = convert_to_ec3(y_data, dass_x_full["MolWeight"].values)
    y_label = np.array([llna_label(v, 2) for v in y_reconverted], dtype=int)
    selected_mask = fit_global_boruta(
        x_full=dass_x_full,
        y_data=y_data,
        boruta_perc=args.boruta_perc,
    )

    # Notebook版と特徴量集合を一致させる
    if "FM percent reacted" in dass_x_full.columns:
        idx = list(dass_x_full.columns).index("FM percent reacted")
        selected_mask[idx] = False

    selected_columns = [
        c for c, flag in zip(dass_x_full.columns, selected_mask)
        if bool(flag)
    ]

    print("\n=== Selected Features ===")
    print(f"Count: {len(selected_columns)}")
    for col in selected_columns:
        print(col)
    print("=========================\n")
    

    config = {
        "Data_split_random_state": args.random_state,
        "Preprocess_pipeline": "Pipeline(KNNImputer(n_neighbors=5, weights='uniform') + StandardScaler())",
        "Feature_selection": True,
        "Feature_selection_function": f"BorutaPy(RandomForestRegressor(max_depth=5), perc={args.boruta_perc}, random_state=0)",
        "Parameter_tuning": True,
        "CV_folds": args.cv_folds,
        "AD_KNN_k": args.ad_knn_k,
        "AD_coverage": args.ad_coverage,
    }

    outer_skf = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    folds: dict[str, Any] = {}
    for fold_no, (tr_idx, te_idx) in enumerate(outer_skf.split(dass_x_full, y_label), start=1):
        x_train_raw = dass_x_full.iloc[tr_idx, :]
        x_test_raw = dass_x_full.iloc[te_idx, :]

        y_train = y_data[tr_idx]
        y_test = y_data[te_idx]

        y_train_ec3 = y_reconverted[tr_idx]
        y_test_ec3 = y_reconverted[te_idx]

        y_train_label = y_label[tr_idx]
        y_test_label = y_label[te_idx]

        mw_train = x_train_raw["MolWeight"].to_numpy()
        mw_test = x_test_raw["MolWeight"].to_numpy()

        pipe = make_pipeline()
        x_train_pre = pipe.fit_transform(x_train_raw)
        x_test_pre = pipe.transform(x_test_raw)

        x_train_sel = np.asarray(x_train_pre)[:, selected_mask]
        x_test_sel = np.asarray(x_test_pre)[:, selected_mask]

        tune_result = optuna_tune_xgb(
            x_train_selected=x_train_sel,
            y_train=y_train,
            y_train_label=y_train_label,
            n_trials=args.n_trials,
        )
        best_params = tune_result["best_params"]

        model = XGBRegressor(**best_params)
        model.fit(x_train_sel, y_train)

        feature_importance = pd.DataFrame(
            list(model.feature_importances_),
            index=dass_x_full.iloc[:, selected_mask].columns.tolist(),
            columns=["importance"],
        ).sort_values("importance", ascending=False)

        fold_dict: dict[str, Any] = {
            "pipeline": pipe,
            "best_params": copy.deepcopy(best_params),
            "model": model,
            "feature_importance": feature_importance,
            "train_selected_X": x_train_sel,
            "train_indices": tr_idx,
            "test_indices": te_idx,
            "metrics": {
                "train_r2": float(r2_score(y_train, model.predict(x_train_sel))),
                "test_r2": float(r2_score(y_test, model.predict(x_test_sel))),
                "train_rmse": float(math.sqrt(mean_squared_error(y_train, model.predict(x_train_sel)))),
                "test_rmse": float(math.sqrt(mean_squared_error(y_test, model.predict(x_test_sel)))),
            },
            "targets": {
                "y_train": y_train,
                "y_test": y_test,
                "ec3_train": y_train_ec3,
                "ec3_test": y_test_ec3,
                "label_train": y_train_label,
                "label_test": y_test_label,
                "mw_train": mw_train,
                "mw_test": mw_test,
            },
        }

        if exval_x_full is not None and exval_ec3 is not None:
            x_exval_pre = pipe.transform(exval_x_full)
            x_exval_sel = np.asarray(x_exval_pre)[:, selected_mask]
            y_exval = convert_to_y(exval_ec3, exval_x_full["MolWeight"].values)

            pred_exval_y = model.predict(x_exval_sel)
            pred_exval_ec3 = convert_to_ec3(pred_exval_y, exval_x_full["MolWeight"].values)
            ad_mask = compute_ad_mask(
                x_new_selected=x_exval_sel,
                x_train_selected=x_train_sel,
                knn_k=args.ad_knn_k,
                ad_coverage=args.ad_coverage,
            )

            fold_dict["exval"] = {
                "pred_y": pred_exval_y,
                "pred_ec3": pred_exval_ec3,
                "true_y": y_exval,
                "true_ec3": exval_ec3.to_numpy(),
                "ad": ad_mask,
            }

        folds[f"fold_{fold_no}"] = fold_dict

        if args.save_shap:
            selected_columns = dass_x_full.iloc[:, selected_mask].columns.tolist()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_train_sel)

            shap_dir = Path(args.output_dir) / "SHAP" / "summary_plot"
            shap_dir.mkdir(parents=True, exist_ok=True)

            plt.figure()
            shap.summary_plot(
                shap_values,
                x_train_sel,
                feature_names=selected_columns,
                plot_type="violin",
                max_display=len(selected_columns),
                show=False,
            )
            plt.savefig(shap_dir / f"summary_plot_fold_{fold_no}.jpg", bbox_inches="tight")
            plt.close()

    artifact = build_artifact(
        full_feature_columns=list(dass_x_full.columns),
        selected_mask=selected_mask,
        folds=folds,
        config=config,
    )

    artifact_path = Path(args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    return artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dass_csv", required=True, help="Notebookで使った DASS CSV")
    parser.add_argument(
        "--exval_csv",
        default=None,
        help="Notebookで使った newCE CSV。未指定なら外部検証予測はスキップ。",
    )
    parser.add_argument(
        "--artifact_path",
        default="artifacts/original_model_artifact.joblib",
        help="保存先 joblib",
    )
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--encoding", default="cp932")
    parser.add_argument("--random_state", type=int, default=1037)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--boruta_perc", type=int, default=100)
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--ad_knn_k", type=int, default=5)
    parser.add_argument("--ad_coverage", type=float, default=0.0)
    parser.add_argument("--save_shap", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved = train_and_export(args)
    print(f"saved: {saved}")