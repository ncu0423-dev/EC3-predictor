from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def build_original_artifact(best_xgbmodel: Any) -> dict[str, Any]:
    """
    Notebook内で構築済みの best_xgbmodel から、
    推論用に安定した辞書アーティファクトを作る。
    """
    full_feature_columns = list(best_xgbmodel._traintest_df_org["X_data"].columns)
    selected_mask = np.asarray(best_xgbmodel._feature_selected_fold_flag, dtype=bool)
    selected_feature_columns = [
        col for col, flag in zip(full_feature_columns, selected_mask) if bool(flag)
    ]

    folds: dict[str, Any] = {}
    for fold_name, fold_dict in best_xgbmodel._cv_fold_data.items():
        folds[fold_name] = {
            "pipeline": copy.deepcopy(fold_dict["Pipeline"]),
            "best_params": copy.deepcopy(fold_dict["Best_params"]),
            "model": copy.deepcopy(fold_dict["Studied_model"]),
            "feature_importance": copy.deepcopy(fold_dict["Feature_importance"]),
            # AD再計算や推論時のfold内参照用
            "train_selected_X": np.asarray(fold_dict["Train_data"]["X_data"], dtype=float),
            "train_indices": np.asarray(fold_dict["Train_test_index"]["Train"]),
            "test_indices": np.asarray(fold_dict["Train_test_index"]["Test"]),
        }

    artifact = {
        "artifact_version": 1,
        "source": {
            "notebook_object_name": "best_xgbmodel",
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "target": {
            "raw_target_column": "LLNA_EC3",
            "y_definition": "log10(MolWeight / EC3)",
            "ec3_inverse_definition": "MolWeight / (10 ** y)",
        },
        "config": copy.deepcopy(best_xgbmodel._config),
        "xgboost_base_params": copy.deepcopy(best_xgbmodel._xgboost_params),
        "full_feature_columns": full_feature_columns,
        "selected_feature_mask": selected_mask.tolist(),
        "selected_feature_columns": selected_feature_columns,
        "folds": folds,
        "ad_config": {
            "knn_k": best_xgbmodel._config.get("AD_KNN_k"),
            "AD_coverage": best_xgbmodel._config.get("AD_coverage"),
        },
    }
    return artifact


def export_best_xgbmodel_to_artifact(
    best_xgbmodel: Any,
    output_path: str | Path = "artifacts/original_model_artifact.joblib",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = build_original_artifact(best_xgbmodel)
    joblib.dump(artifact, output_path, compress=3)

    return output_path


if __name__ == "__main__":
    # Notebook上で次のどちらかで実行する想定:
    # 1) from save_original_artifact import export_best_xgbmodel_to_artifact
    #    export_best_xgbmodel_to_artifact(best_xgbmodel)
    # 2) %run -i save_original_artifact.py
    if "best_xgbmodel" not in globals():
        raise RuntimeError(
            "best_xgbmodel が見つかりません。Notebookで学習済みの best_xgbmodel を作成した後に、"
            "このファイルを import して export_best_xgbmodel_to_artifact(best_xgbmodel) を呼んでください。"
        )

    saved_path = export_best_xgbmodel_to_artifact(globals()["best_xgbmodel"])
    print(f"saved: {saved_path}")