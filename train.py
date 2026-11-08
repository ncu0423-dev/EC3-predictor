# -*- coding: utf-8 -*-
"""
Notebook再現用 学習スクリプト（EC3最小fold採用版）

主な仕様
- KNNImputer + StandardScaler
- Boruta または Notebook固定特徴量
- XGBoost + 任意のOptunaチューニング
- Stratified 5-fold CV
- 外部検証(newCE)は「各行で予測EC3が最小のfold」を代表foldとして採用
- k-NN ADを考慮する場合は「AD内foldのうち予測EC3が最小」を採用
- SHAP保存
- Streamlit/runtime用 artifact(joblib) 保存

例:
    python train.py \
      --train-csv DASS_join_data_opera_delete_LLNAnan.csv \
      --exval-csv newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv \
      --artifact-path artifacts/original_model_artifact.joblib \
      --output-dir outputs \
      --feature-mode fixed \
      --ad-coverage 0.1
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils_io import (
    MW_COL,
    NOTEBOOK_SELECTED_FEATURE_COLUMNS,
    TARGET_COLUMN_DEFAULT,
    ensure_dir,
    load_dataset,
    save_json,
)

from utils_ml import (
    NOTEBOOK_XGB_BASE_PARAMS,
    build_notebook_style_test_output,
    clone_and_transform,
    compute_knn_ad,
    convert_ec3_to_y,
    convert_y_to_ec3,
    extract_invitro_calls,
    fit_global_boruta,
    make_llna_labels,
    make_preprocess_pipeline,
    optimize_xgb_params,
    save_shap_artifacts_for_fold,
    valid_regression_metrics,
    _fit_model,
)


TARGET_OUTPUT_COL = "LLNA EC3(NS converted to 150%)"
META_COLUMNS = ("No.", "Name", "CAS Number", "SMILES")


def _as_1d_float(
    values,
    *,
    name: str,
    expected_len: int | None = None,
) -> np.ndarray:
    """値を1次元float配列へ安全に変換する。"""
    arr = np.asarray(values, dtype=float).reshape(-1)

    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(
            f"{name} の行数が不正です: "
            f"expected={expected_len}, actual={len(arr)}"
        )

    return arr


def _validate_molecular_weight(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
) -> np.ndarray:
    """
    EC3逆変換に必要なMolWeightを検証する。
    欠損・非数値・0以下は許可しない。
    """
    if MW_COL not in frame.columns:
        raise KeyError(
            f"{dataset_name} に {MW_COL} 列がありません。"
        )

    mw = pd.to_numeric(
        frame[MW_COL],
        errors="coerce",
    ).to_numpy(dtype=float)

    invalid = ~np.isfinite(mw) | (mw <= 0)

    if invalid.any():
        rows = np.flatnonzero(invalid)[:10].tolist()

        raise ValueError(
            f"{dataset_name} の {MW_COL} に"
            f"欠損・非数値・0以下があります。"
            f" 該当行(先頭10件): {rows}"
        )

    return mw


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Notebook形式の Fold_misprediction:
        predicted_EC3 / true_EC3
    """
    num = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    den = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        out = num / den

    return out.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _select_min_ec3_fold(
    pred_y_fold: pd.DataFrame,
    pred_ec3_fold: pd.DataFrame,
    *,
    eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    各行で Predict_LLNA EC3 が最小のfoldを選ぶ。

    eligibility が指定された場合:
        True のfoldだけを候補にする。

    重要:
        EC3だけmin()してpred_yを別foldから取るのではなく、
        EC3が最小になった「同じfold」のpred_yを採用する。
    """

    if pred_ec3_fold.empty:
        raise ValueError(
            "fold予測が0列です。"
        )

    columns = list(
        pred_ec3_fold.columns
    )

    if list(pred_y_fold.columns) != columns:
        raise ValueError(
            "pred_y_fold と pred_ec3_fold の"
            "fold列が一致していません。"
        )

    if len(pred_y_fold) != len(pred_ec3_fold):
        raise ValueError(
            "pred_y_fold と pred_ec3_fold の"
            "行数が一致していません。"
        )

    y_values = (
        pred_y_fold
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    ec3_values = (
        pred_ec3_fold
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(y_values)
        & np.isfinite(ec3_values)
        & (ec3_values > 0)
    )

    if eligibility is not None:

        eligibility = eligibility.reindex(
            index=pred_ec3_fold.index,
            columns=columns,
        )

        eligible_values = (
            eligibility
            .fillna(False)
            .astype(bool)
            .to_numpy(dtype=bool)
        )

        valid &= eligible_values

    # 無効値は inf にして
    # argmin の候補から外す
    safe_ec3 = np.where(
        valid,
        ec3_values,
        np.inf,
    )

    chosen_idx = np.argmin(
        safe_ec3,
        axis=1,
    )

    has_candidate = valid.any(
        axis=1
    )

    row_idx = np.arange(
        len(pred_ec3_fold)
    )

    chosen_y = np.full(
        len(pred_ec3_fold),
        np.nan,
        dtype=float,
    )

    chosen_ec3 = np.full(
        len(pred_ec3_fold),
        np.nan,
        dtype=float,
    )

    chosen_fold = np.full(
        len(pred_ec3_fold),
        None,
        dtype=object,
    )

    chosen_y[has_candidate] = y_values[
        row_idx[has_candidate],
        chosen_idx[has_candidate],
    ]

    chosen_ec3[has_candidate] = ec3_values[
        row_idx[has_candidate],
        chosen_idx[has_candidate],
    ]

    col_array = np.asarray(
        columns,
        dtype=object,
    )

    chosen_fold[has_candidate] = col_array[
        chosen_idx[has_candidate]
    ]

    return pd.DataFrame(
        {
            "selected_fold": chosen_fold,
            "selected_fold_index": np.where(
                has_candidate,
                chosen_idx,
                -1,
            ),
            "pred_y": chosen_y,
            "pred_ec3": chosen_ec3,
            "has_candidate": has_candidate,
        },
        index=pred_ec3_fold.index,
    )


def _selected_bool_from_fold(
    bool_fold: pd.DataFrame,
    selection: pd.DataFrame,
) -> np.ndarray:
    """
    選択されたfoldに対応する
    AD bool値を取り出す。
    """

    values = (
        bool_fold
        .fillna(False)
        .astype(bool)
        .to_numpy(dtype=bool)
    )

    idx = selection[
        "selected_fold_index"
    ].to_numpy(dtype=int)

    has = selection[
        "has_candidate"
    ].to_numpy(dtype=bool)

    rows = np.arange(
        len(bool_fold)
    )

    out = np.zeros(
        len(bool_fold),
        dtype=bool,
    )

    out[has] = values[
        rows[has],
        idx[has],
    ]

    return out


def _build_min_ec3_exval_output(
    *,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_true_ec3: np.ndarray,
    pred_y_fold: pd.DataFrame,
    pred_ec3_fold: pd.DataFrame,
    knn_ad_fold: pd.DataFrame,
    X_original: pd.DataFrame,
) -> pd.DataFrame:
    """
    外部検証結果を
    「EC3最小fold」の規則で作成する。
    """

    n = len(
        pred_ec3_fold
    )

    y_true = _as_1d_float(
        y_true,
        name="y_true",
        expected_len=n,
    )

    y_true_ec3 = _as_1d_float(
        y_true_ec3,
        name="y_true_ec3",
        expected_len=n,
    )

    out = pd.DataFrame(
        index=np.arange(n)
    )

    meta_cols = [
        c
        for c in META_COLUMNS
        if c in meta_df.columns
    ]

    if meta_cols:
        out = pd.concat(
            [
                meta_df[
                    meta_cols
                ].reset_index(drop=True),
                out,
            ],
            axis=1,
        )

    out["true_y"] = y_true

    out[
        TARGET_OUTPUT_COL
    ] = y_true_ec3

    # ==================================================
    # 全foldからEC3最小foldを選択
    # ==================================================
    selected = _select_min_ec3_fold(
        pred_y_fold,
        pred_ec3_fold,
    )

    out["selected_fold"] = (
        selected[
            "selected_fold"
        ]
        .reset_index(drop=True)
    )

    out["pred_y"] = selected[
        "pred_y"
    ].to_numpy(dtype=float)

    out["Predict_LLNA EC3"] = selected[
        "pred_ec3"
    ].to_numpy(dtype=float)

    out["Fold_misprediction"] = _safe_ratio(
        out["Predict_LLNA EC3"],
        out[TARGET_OUTPUT_COL],
    )

    # 選択された代表foldがAD内か
    out["kNN_AD"] = _selected_bool_from_fold(
        knn_ad_fold,
        selected,
    )

    # 5fold中1つでもAD内ならTrue
    out["any_kNN_AD"] = (
        knn_ad_fold
        .fillna(False)
        .astype(bool)
        .any(axis=1)
        .to_numpy()
    )

    # ==================================================
    # k-NN AD内のfoldだけから
    # EC3最小foldを選択
    # ==================================================
    selected_in_ad = _select_min_ec3_fold(
        pred_y_fold,
        pred_ec3_fold,
        eligibility=knn_ad_fold,
    )

    out["knn_inAD_selected_fold"] = (
        selected_in_ad[
            "selected_fold"
        ]
        .reset_index(drop=True)
    )

    out["knn_inAD_pred_y"] = (
        selected_in_ad[
            "pred_y"
        ]
        .to_numpy(dtype=float)
    )

    out[
        "knn_inAD_Predict_LLNA EC3"
    ] = (
        selected_in_ad[
            "pred_ec3"
        ]
        .to_numpy(dtype=float)
    )

    out[
        "knn_inAD_Fold_misprediction"
    ] = _safe_ratio(
        out[
            "knn_inAD_Predict_LLNA EC3"
        ],
        out[TARGET_OUTPUT_COL],
    )

    # ==================================================
    # in vitro AD
    # ==================================================
    try:
        invitro_df = extract_invitro_calls(
            X_original
        )

        out["invitro_AD"] = (
            invitro_df[
                "invitro_AD"
            ]
            .reset_index(drop=True)
        )

        out["vitro_call_sum"] = (
            invitro_df[
                "vitro_call_sum"
            ]
            .reset_index(drop=True)
        )

    except Exception as exc:

        print(
            "[WARN] "
            "in vitro ADを計算できませんでした:",
            exc,
        )

        out["invitro_AD"] = pd.Series(
            [pd.NA] * n,
            dtype="boolean",
        )

        out["vitro_call_sum"] = np.nan

    # ==================================================
    # fold別予測をすべて保存
    # ==================================================
    out = pd.concat(
        [
            out.reset_index(drop=True),

            knn_ad_fold
            .add_prefix("knn_AD_")
            .reset_index(drop=True),

            pred_y_fold
            .add_prefix("pred_y_")
            .reset_index(drop=True),

            pred_ec3_fold
            .add_prefix(
                "Predict_LLNA EC3_"
            )
            .reset_index(drop=True),
        ],
        axis=1,
    )

    out["pred_y_mean"] = (
        pred_y_fold
        .mean(
            axis=1,
            skipna=True,
        )
        .to_numpy()
    )

    out[
        "Predict_LLNA EC3_mean"
    ] = (
        pred_ec3_fold
        .mean(
            axis=1,
            skipna=True,
        )
        .to_numpy()
    )

    out["pred_y_max"] = (
        pred_y_fold
        .max(
            axis=1,
            skipna=True,
        )
        .to_numpy()
    )

    out[
        "Predict_LLNA EC3_min"
    ] = (
        pred_ec3_fold
        .min(
            axis=1,
            skipna=True,
        )
        .to_numpy()
    )

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Notebook再現版 "
            "XGBoost 学習スクリプト"
        )
    )

    p.add_argument(
        "--train-csv",
        required=True,
        help="DASS学習CSV",
    )

    p.add_argument(
        "--exval-csv",
        required=True,
        help="newCE外部検証CSV",
    )

    p.add_argument(
        "--target-col",
        default=TARGET_COLUMN_DEFAULT,
        help="目的変数列名",
    )

    p.add_argument(
        "--artifact-path",
        default=(
            "artifacts/"
            "original_model_artifact.joblib"
        ),
    )

    p.add_argument(
        "--output-dir",
        default="outputs",
    )

    p.add_argument(
        "--feature-mode",
        choices=[
            "boruta",
            "fixed",
        ],
        default="boruta",
        help=(
            "boruta=Boruta再計算 / "
            "fixed=Notebook固定特徴量"
        ),
    )

    p.add_argument(
        "--random-state",
        type=int,
        default=1037,
    )

    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )

    p.add_argument(
        "--parameter-tuning",
        action="store_true",
    )

    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
    )

    p.add_argument(
        "--boruta-rf-depth",
        type=int,
        default=5,
    )

    p.add_argument(
        "--boruta-perc",
        type=int,
        default=100,
    )

    p.add_argument(
        "--knn-neighbors",
        type=int,
        default=5,
    )

    p.add_argument(
        "--knn-weights",
        default="uniform",
    )

    p.add_argument(
        "--ad-k",
        type=int,
        default=5,
    )

    p.add_argument(
        "--ad-coverage",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--save-force-plots",
        action="store_true",
    )

    p.add_argument(
        "--skip-shap",
        action="store_true",
    )

    return p.parse_args()


def main() -> None:

    args = parse_args()

    # ==================================================
    # 引数チェック
    # ==================================================
    if args.cv_folds < 2:
        raise ValueError(
            "--cv-folds は2以上にしてください。"
        )

    if args.ad_k < 1:
        raise ValueError(
            "--ad-k は1以上にしてください。"
        )

    if not (
        0.0 <= args.ad_coverage < 1.0
    ):
        raise ValueError(
            "--ad-coverage は"
            "0以上1未満にしてください。"
        )

    # ==================================================
    # 出力ディレクトリ
    # ==================================================
    output_dir = Path(
        args.output_dir
    )

    reports_dir = ensure_dir(
        output_dir / "reports"
    )

    shap_dir = ensure_dir(
        reports_dir / "SHAP"
    )

    shap_summary_dir = ensure_dir(
        shap_dir / "summary_plot"
    )

    shap_force_dir = ensure_dir(
        shap_dir / "force_plot"
    )

    pred_dir = ensure_dir(
        output_dir / "prediction_report"
    )

    debug_dir = ensure_dir(
        output_dir / "debug"
    )

    print(
        "[INFO] train_csv:",
        args.train_csv,
    )

    print(
        "[INFO] exval_csv:",
        args.exval_csv,
    )

    print(
        "[INFO] aggregation:"
        " minimum EC3 per row"
    )

    # ==================================================
    # データ読込
    # ==================================================
    train_ds = load_dataset(
        args.train_csv,
        target_col=args.target_col,
    )

    exval_ds = load_dataset(
        args.exval_csv,
        target_col=args.target_col,
    )

    X_train_full: pd.DataFrame = (
        train_ds[
            "X_full"
        ].copy()
    )

    X_exval_full: pd.DataFrame = (
        exval_ds[
            "X_full"
        ].copy()
    )

    y_train_ec3 = pd.to_numeric(
        train_ds["y_ec3"],
        errors="coerce",
    )

    y_exval_ec3 = pd.to_numeric(
        exval_ds["y_ec3"],
        errors="coerce",
    )

    meta_train = (
        train_ds[
            "meta"
        ].copy()
    )

    meta_exval = (
        exval_ds[
            "meta"
        ].copy()
    )

    # ==================================================
    # exval列を学習時と同じ順序にそろえる
    # ==================================================
    missing_exval_full = [
        c
        for c in X_train_full.columns
        if c not in X_exval_full.columns
    ]

    if missing_exval_full:
        raise KeyError(
            "exval CSVに学習特徴量が"
            f"不足しています: {missing_exval_full}"
        )

    X_exval_full = (
        X_exval_full
        .loc[
            :,
            X_train_full.columns,
        ]
        .copy()
    )

    # ==================================================
    # MolWeight検証
    # ==================================================
    mw_train = _validate_molecular_weight(
        X_train_full,
        dataset_name="train",
    )

    mw_exval = _validate_molecular_weight(
        X_exval_full,
        dataset_name="exval",
    )

    train_ec3_values = (
        y_train_ec3
        .to_numpy(dtype=float)
    )

    exval_ec3_values = (
        y_exval_ec3
        .to_numpy(dtype=float)
    )

    if (
        ~np.isfinite(train_ec3_values)
        | (train_ec3_values <= 0)
    ).any():
        raise ValueError(
            "trainの目的変数EC3に"
            "欠損・非数値・0以下があります。"
        )

    if (
        ~np.isfinite(exval_ec3_values)
        | (exval_ec3_values <= 0)
    ).any():
        raise ValueError(
            "exvalの目的変数EC3に"
            "欠損・非数値・0以下があります。"
        )

    # ==================================================
    # EC3 -> y
    # ==================================================
    y_train = _as_1d_float(
        convert_ec3_to_y(
            y_train_ec3,
            mw_train,
        ),
        name="y_train",
        expected_len=len(
            X_train_full
        ),
    )

    y_exval = _as_1d_float(
        convert_ec3_to_y(
            y_exval_ec3,
            mw_exval,
        ),
        name="y_exval",
        expected_len=len(
            X_exval_full
        ),
    )

    debug_nan = {
        "train_rows":
            int(
                len(X_train_full)
            ),
        "exval_rows":
            int(
                len(X_exval_full)
            ),
        "train_target_nan":
            int(
                pd.isna(
                    y_train_ec3
                ).sum()
            ),
        "exval_target_nan":
            int(
                pd.isna(
                    y_exval_ec3
                ).sum()
            ),
        "train_mw_nan":
            int(
                pd.isna(
                    mw_train
                ).sum()
            ),
        "exval_mw_nan":
            int(
                pd.isna(
                    mw_exval
                ).sum()
            ),
        "train_y_nan":
            int(
                np.isnan(
                    y_train
                ).sum()
            ),
        "exval_y_nan":
            int(
                np.isnan(
                    y_exval
                ).sum()
            ),
    }

    save_json(
        debug_nan,
        debug_dir
        / "nan_summary.json",
    )

    print(
        "[DEBUG] nan_summary:",
        json.dumps(
            debug_nan,
            ensure_ascii=False,
        ),
    )

    # ==================================================
    # StratifiedKFold用ラベル
    # ==================================================
    labels_train = make_llna_labels(
        train_ec3_values,
        boundary=2.0,
    )

    preprocess_pipeline = (
        make_preprocess_pipeline(
            n_neighbors=(
                args.knn_neighbors
            ),
            weights=(
                args.knn_weights
            ),
        )
    )

    # ==================================================
    # 特徴量選択
    # ==================================================
    if args.feature_mode == "boruta":

        (
            selected_mask,
            selected_columns,
            _boruta,
        ) = fit_global_boruta(
            X_train_full,
            y_train,
            preprocess_pipeline,
            rf_depth=(
                args.boruta_rf_depth
            ),
            perc=(
                args.boruta_perc
            ),
            random_state=0,
        )

        selected_columns = list(
            selected_columns
        )

    else:

        selected_columns = [
            c
            for c
            in NOTEBOOK_SELECTED_FEATURE_COLUMNS
            if c
            in X_train_full.columns
        ]

    # Notebook互換
    selected_columns = [
        c
        for c in selected_columns
        if c
        != "FM percent reacted"
    ]

    # 除外後にmaskを必ず作り直す
    selected_set = set(
        selected_columns
    )

    selected_mask = np.array(
        [
            c in selected_set
            for c
            in X_train_full.columns
        ],
        dtype=bool,
    )

    if (
        not selected_columns
        or not selected_mask.any()
    ):
        raise ValueError(
            "選択特徴量が0件です。"
        )

    if (
        int(selected_mask.sum())
        != len(selected_columns)
    ):
        raise ValueError(
            "selected_feature_columns と "
            "selected_feature_mask が"
            "不整合です。"
        )

    pd.DataFrame(
        {
            "full_feature_columns":
                X_train_full.columns,

            "selected":
                selected_mask.astype(int),
        }
    ).to_csv(
        debug_dir
        / "selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[INFO] selected_feature_count:",
        int(
            selected_mask.sum()
        ),
    )

    print(
        "[INFO] selected_feature_columns:",
        selected_columns,
    )

    # ==================================================
    # CV
    # ==================================================
    skf = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    base_params = deepcopy(
        NOTEBOOK_XGB_BASE_PARAMS
    )

    folds: list[dict] = []

    test_pred_y_all = np.full(
        len(X_train_full),
        np.nan,
        dtype=float,
    )

    test_pred_ec3_all = np.full(
        len(X_train_full),
        np.nan,
        dtype=float,
    )

    test_ad_all = np.full(
        len(X_train_full),
        False,
        dtype=bool,
    )

    exval_pred_y_fold = pd.DataFrame(
        index=X_exval_full.index
    )

    exval_pred_ec3_fold = pd.DataFrame(
        index=X_exval_full.index
    )

    exval_ad_fold = pd.DataFrame(
        index=X_exval_full.index
    )

    exval_metrics_list: list[dict] = []

    # ==================================================
    # fold学習
    # ==================================================
    for fold_no, (
        tr_idx,
        te_idx,
    ) in enumerate(
        skf.split(
            X_train_full,
            labels_train,
        ),
        start=1,
    ):

        fold_name = (
            f"fold_{fold_no}"
        )

        print(
            f"[TRAIN] {fold_name}: "
            f"train={len(tr_idx)} "
            f"test={len(te_idx)}"
        )

        X_cvtrain = (
            X_train_full
            .iloc[tr_idx]
            .copy()
        )

        X_cvtest = (
            X_train_full
            .iloc[te_idx]
            .copy()
        )

        y_cvtrain = (
            y_train[tr_idx]
        )

        y_cvtest = (
            y_train[te_idx]
        )

        labels_cvtrain = (
            labels_train[tr_idx]
        )

        mw_cvtrain = (
            mw_train[tr_idx]
        )

        mw_cvtest = (
            mw_train[te_idx]
        )

        # ==============================================
        # 非常に重要
        # 1 foldにつきpipelineは1回だけfit
        # 同じpipeでtest/exvalをtransform
        # ==============================================
        (
            X_tr_pre,
            X_te_pre,
            pipe,
        ) = clone_and_transform(
            preprocess_pipeline,
            X_cvtrain,
            X_cvtest,
        )

        X_ex_pre = pipe.transform(
            X_exval_full
        )

        X_tr_sel = (
            np.asarray(
                X_tr_pre,
                dtype=float,
            )[:, selected_mask]
        )

        X_te_sel = (
            np.asarray(
                X_te_pre,
                dtype=float,
            )[:, selected_mask]
        )

        X_ex_sel = (
            np.asarray(
                X_ex_pre,
                dtype=float,
            )[:, selected_mask]
        )

        # ==============================================
        # Parameter tuning
        # ==============================================
        if args.parameter_tuning:

            (
                best_params,
                study,
                _best_iters,
                best_n_estimators,
            ) = optimize_xgb_params(
                X_tr_sel,
                y_cvtrain,
                labels_cvtrain,
                base_params=base_params,
                n_trials=args.n_trials,
                random_state=0,
            )

            study_best_value = float(
                study.best_value
            )

            study_best_trial = int(
                study.best_trial.number
            )

        else:

            best_params = deepcopy(
                base_params
            )

            best_params[
                "n_estimators"
            ] = 150

            best_params.pop(
                "early_stopping_rounds",
                None,
            )

            best_n_estimators = 150

            study_best_value = float(
                "nan"
            )

            study_best_trial = -1

        # ==============================================
        # モデル学習
        # ==============================================
        model = _fit_model(
            best_params,
            X_tr_sel,
            y_cvtrain,
        )

        # ==============================================
        # y予測
        # ==============================================
        pred_tr_y = _as_1d_float(
            model.predict(
                X_tr_sel
            ),
            name=(
                f"{fold_name}."
                "pred_tr_y"
            ),
            expected_len=len(
                tr_idx
            ),
        )

        pred_te_y = _as_1d_float(
            model.predict(
                X_te_sel
            ),
            name=(
                f"{fold_name}."
                "pred_te_y"
            ),
            expected_len=len(
                te_idx
            ),
        )

        pred_ev_y = _as_1d_float(
            model.predict(
                X_ex_sel
            ),
            name=(
                f"{fold_name}."
                "pred_ev_y"
            ),
            expected_len=len(
                X_exval_full
            ),
        )

        # ==============================================
        # y -> EC3
        # ==============================================
        pred_tr_ec3 = _as_1d_float(
            convert_y_to_ec3(
                pred_tr_y,
                mw_cvtrain,
            ),
            name=(
                f"{fold_name}."
                "pred_tr_ec3"
            ),
            expected_len=len(
                tr_idx
            ),
        )

        pred_te_ec3 = _as_1d_float(
            convert_y_to_ec3(
                pred_te_y,
                mw_cvtest,
            ),
            name=(
                f"{fold_name}."
                "pred_te_ec3"
            ),
            expected_len=len(
                te_idx
            ),
        )

        pred_ev_ec3 = _as_1d_float(
            convert_y_to_ec3(
                pred_ev_y,
                mw_exval,
            ),
            name=(
                f"{fold_name}."
                "pred_ev_ec3"
            ),
            expected_len=len(
                X_exval_full
            ),
        )

        if (
            ~np.isfinite(
                pred_ev_ec3
            )
            | (
                pred_ev_ec3 <= 0
            )
        ).all():

            raise ValueError(
                f"{fold_name}: "
                "外部検証EC3が"
                "全件無効です。"
            )

        # ==============================================
        # AD
        # ==============================================
        (
            y_test_ad,
            y_test_knn_dist,
            ad_threshold,
        ) = compute_knn_ad(
            X_tr_sel,
            X_te_sel,
            knn_k=args.ad_k,
            coverage=args.ad_coverage,
        )

        (
            y_exval_ad,
            y_ex_knn_dist,
            _,
        ) = compute_knn_ad(
            X_tr_sel,
            X_ex_sel,
            knn_k=args.ad_k,
            coverage=args.ad_coverage,
        )

        y_test_ad = np.asarray(
            y_test_ad,
            dtype=bool,
        ).reshape(-1)

        y_exval_ad = np.asarray(
            y_exval_ad,
            dtype=bool,
        ).reshape(-1)

        # CV testは各行1つのheld-out fold
        test_pred_y_all[
            te_idx
        ] = pred_te_y

        test_pred_ec3_all[
            te_idx
        ] = pred_te_ec3

        test_ad_all[
            te_idx
        ] = y_test_ad

        # 外部検証は全fold保持
        exval_pred_y_fold[
            fold_name
        ] = pred_ev_y

        exval_pred_ec3_fold[
            fold_name
        ] = pred_ev_ec3

        exval_ad_fold[
            fold_name
        ] = y_exval_ad

        # ==============================================
        # SHAP
        # ==============================================
        shap_paths: dict = {}

        if not args.skip_shap:

            try:
                shap_paths = (
                    save_shap_artifacts_for_fold(
                        model,
                        X_tr_sel,
                        selected_columns,
                        fold_name=fold_name,
                        output_dir=(
                            shap_summary_dir
                        ),
                        save_force=False,
                    )
                )

                if args.save_force_plots:

                    force_paths = (
                        save_shap_artifacts_for_fold(
                            model,
                            X_tr_sel,
                            selected_columns,
                            fold_name=fold_name,
                            output_dir=(
                                shap_force_dir
                            ),
                            save_force=True,
                        )
                    )

                    if force_paths.get(
                        "force_plot_html"
                    ):
                        shap_paths[
                            "force_plot_html"
                        ] = force_paths[
                            "force_plot_html"
                        ]

            except Exception as exc:

                print(
                    f"[WARN] {fold_name}: "
                    "SHAP保存に失敗:",
                    exc,
                )

                shap_paths = {
                    "error": str(exc)
                }

        # ==============================================
        # metrics
        # ==============================================
        train_metrics = (
            valid_regression_metrics(
                y_cvtrain,
                pred_tr_y,
            )
        )

        test_metrics = (
            valid_regression_metrics(
                y_cvtest,
                pred_te_y,
            )
        )

        exval_metrics = (
            valid_regression_metrics(
                y_exval,
                pred_ev_y,
            )
        )

        exval_metrics_list.append(
            exval_metrics
        )

        # ==============================================
        # artifact fold
        # ==============================================
        fold_artifact = {
            "fold_name":
                fold_name,

            "train_index":
                tr_idx.tolist(),

            "test_index":
                te_idx.tolist(),

            "pipeline":
                pipe,

            "model":
                model,

            "best_params":
                best_params,

            "parameter_tuning":
                bool(
                    args.parameter_tuning
                ),

            "optuna_best_value":
                study_best_value,

            "optuna_best_trial_number":
                study_best_trial,

            "best_n_estimators":
                int(
                    best_n_estimators
                ),

            "selected_feature_columns":
                list(
                    selected_columns
                ),

            "selected_feature_mask":
                selected_mask.tolist(),

            "X_train_selected":
                X_tr_sel,

            "train_metrics":
                train_metrics,

            "test_metrics":
                test_metrics,

            "exval_metrics":
                exval_metrics,

            "ad_threshold":
                float(
                    ad_threshold
                ),

            "ad_knn_k":
                int(
                    args.ad_k
                ),

            "ad_coverage":
                float(
                    args.ad_coverage
                ),

            "test_knn_dist":
                np.asarray(
                    y_test_knn_dist,
                    dtype=float,
                )
                .reshape(-1)
                .tolist(),

            "exval_knn_dist":
                np.asarray(
                    y_ex_knn_dist,
                    dtype=float,
                )
                .reshape(-1)
                .tolist(),

            "shap_paths":
                shap_paths,
        }

        folds.append(
            fold_artifact
        )

    # ==================================================
    # CV予測欠損チェック
    # ==================================================
    if (
        np.isnan(
            test_pred_y_all
        ).any()
        or
        np.isnan(
            test_pred_ec3_all
        ).any()
    ):
        raise RuntimeError(
            "CVテスト予測に"
            "未代入行があります。"
        )

    # ==================================================
    # DASS出力
    # held-out foldをそのまま使用
    # ==================================================
    test_output_df = (
        build_notebook_style_test_output(
            meta_df=meta_train,
            y_true=y_train,
            y_true_ec3=(
                train_ec3_values
            ),
            pred_y=(
                test_pred_y_all
            ),
            pred_ec3=(
                test_pred_ec3_all
            ),
            knn_ad=(
                test_ad_all
            ),
            X_original=(
                X_train_full
            ),
        )
    )

    # ==================================================
    # newCE出力
    # EC3最小foldを行ごとに選択
    # ==================================================
    exval_output_df = (
        _build_min_ec3_exval_output(
            meta_df=meta_exval,
            y_true=y_exval,
            y_true_ec3=(
                exval_ec3_values
            ),
            pred_y_fold=(
                exval_pred_y_fold
            ),
            pred_ec3_fold=(
                exval_pred_ec3_fold
            ),
            knn_ad_fold=(
                exval_ad_fold
            ),
            X_original=(
                X_exval_full
            ),
        )
    )

    # ==================================================
    # CSV出力
    # ==================================================
    test_output_path = (
        pred_dir
        / "DASS_predict_result_reproduced.csv"
    )

    exval_output_path = (
        pred_dir
        / "newCE_predict_result_reproduced.csv"
    )

    test_output_df.to_csv(
        test_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    exval_output_df.to_csv(
        exval_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ==================================================
    # foldごとの前処理後データ
    # ==================================================
    train_data_dir = ensure_dir(
        output_dir
        / "train_test_data"
    )

    exval_data_dir = ensure_dir(
        output_dir
        / "exval_data"
    )

    for fold in folds:

        fold_name = (
            fold["fold_name"]
        )

        tr_idx = (
            fold["train_index"]
        )

        te_idx = (
            fold["test_index"]
        )

        selected_cols = (
            fold[
                "selected_feature_columns"
            ]
        )

        pipe = (
            fold["pipeline"]
        )

        fold_mask = np.asarray(
            fold[
                "selected_feature_mask"
            ],
            dtype=bool,
        )

        X_fold_train_pre = (
            np.asarray(
                pipe.transform(
                    X_train_full.iloc[
                        tr_idx
                    ]
                ),
                dtype=float,
            )[:, fold_mask]
        )

        X_fold_test_pre = (
            np.asarray(
                pipe.transform(
                    X_train_full.iloc[
                        te_idx
                    ]
                ),
                dtype=float,
            )[:, fold_mask]
        )

        X_fold_exval_pre = (
            np.asarray(
                pipe.transform(
                    X_exval_full
                ),
                dtype=float,
            )[:, fold_mask]
        )

        df_train_fold = pd.concat(
            [
                meta_train
                .iloc[tr_idx]
                .reset_index(
                    drop=True
                ),

                pd.DataFrame(
                    X_fold_train_pre,
                    columns=selected_cols,
                ),
            ],
            axis=1,
        )

        df_test_fold = pd.concat(
            [
                meta_train
                .iloc[te_idx]
                .reset_index(
                    drop=True
                ),

                pd.DataFrame(
                    X_fold_test_pre,
                    columns=selected_cols,
                ),
            ],
            axis=1,
        )

        pd.concat(
            [
                df_train_fold,
                df_test_fold,
            ],
            ignore_index=True,
        ).to_csv(
            train_data_dir
            / (
                "train_test_data_"
                f"{fold_name}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        pd.concat(
            [
                meta_exval
                .reset_index(
                    drop=True
                ),

                pd.DataFrame(
                    X_fold_exval_pre,
                    columns=selected_cols,
                ),
            ],
            axis=1,
        ).to_csv(
            exval_data_dir
            / (
                "exval_data_"
                f"{fold_name}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

    # ==================================================
    # Aggregate metrics
    # ==================================================
    aggregate_metrics = {

        "aggregation_rule":
            "minimum_predicted_ec3_per_row",

        "test_overall":
            valid_regression_metrics(
                y_train,
                test_pred_y_all,
            ),

        "exval_overall_allfold_no_ad":
            valid_regression_metrics(
                y_exval,
                exval_output_df[
                    "pred_y"
                ].to_numpy(
                    dtype=float
                ),
            ),

        "exval_overall_knn_inad":
            valid_regression_metrics(
                y_exval,
                exval_output_df[
                    "knn_inAD_pred_y"
                ].to_numpy(
                    dtype=float
                ),
            ),

        "fold_exval_metrics":
            exval_metrics_list,
    }

    save_json(
        aggregate_metrics,
        reports_dir
        / "metrics.json",
    )

    # ==================================================
    # artifact保存
    # ==================================================
    artifact = {

        "artifact_version":
            "repro_v2_min_ec3",

        "source":
            (
                "XGBoost_model_refact_tokunaga "
                "Notebook + corrected train.py"
            ),

        "target":
            args.target_col,

        "prediction_aggregation":
            "minimum_predicted_ec3_per_row",

        "config": {
            "random_state":
                args.random_state,

            "cv_folds":
                args.cv_folds,

            "parameter_tuning":
                bool(
                    args.parameter_tuning
                ),

            "n_trials":
                int(
                    args.n_trials
                ),

            "feature_mode":
                args.feature_mode,

            "boruta_rf_depth":
                args.boruta_rf_depth,

            "boruta_perc":
                args.boruta_perc,

            "knn_neighbors":
                args.knn_neighbors,

            "knn_weights":
                args.knn_weights,

            "ad_k":
                args.ad_k,

            "ad_coverage":
                args.ad_coverage,
        },

        "xgboost_base_params":
            base_params,

        "full_feature_columns":
            X_train_full.columns.tolist(),

        "selected_feature_mask":
            selected_mask.tolist(),

        "selected_feature_columns":
            list(
                selected_columns
            ),

        # Streamlit/runtime側で
        # MolWeightを必須にできる
        "runtime_required_columns":
            list(
                dict.fromkeys(
                    [
                        MW_COL,
                        *selected_columns,
                    ]
                )
            ),

        "notebook_selected_feature_columns":
            list(
                NOTEBOOK_SELECTED_FEATURE_COLUMNS
            ),

        "folds":
            folds,

        "ad_config": {
            "knn_k":
                args.ad_k,

            "coverage":
                args.ad_coverage,
        },

        "train_csv":
            os.path.abspath(
                args.train_csv
            ),

        "exval_csv":
            os.path.abspath(
                args.exval_csv
            ),

        "train_output_csv":
            str(
                test_output_path
            ),

        "exval_output_csv":
            str(
                exval_output_path
            ),

        "metrics":
            aggregate_metrics,
    }

    artifact_path = Path(
        args.artifact_path
    )

    ensure_dir(
        artifact_path.parent
    )

    joblib.dump(
        artifact,
        artifact_path,
    )

    print(
        "[DONE] artifact saved:",
        artifact_path,
    )

    print(
        "[DONE] test_output:",
        test_output_path,
    )

    print(
        "[DONE] exval_output:",
        exval_output_path,
    )

    print(
        "[DONE] representative prediction"
        " = minimum EC3 fold per row"
    )


if __name__ == "__main__":
    main()
