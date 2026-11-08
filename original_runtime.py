# -*- coding: utf-8 -*-
"""
artifactベース推論ランタイム（EC3最小fold採用版）

仕様
- 学習済み各foldのpipeline/modelをそのまま利用
- 各入力行について、全foldの予測EC3を計算
- Predict_LLNA EC3 は「最小EC3を出したfold」の値を採用
- pred_y も同じfoldの値を採用
- k-NN AD内予測は、AD=TrueのfoldだけからEC3最小を採用
- MolWeightを入力CSVから直接読み、EC3がNaNになる原因を検査
- テンプレートには必ずMolWeightを含める
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from utils_io import (
    MW_COL,
    TARGET_COLUMN_DEFAULT,
    ensure_dir,
    read_csv_auto,
    validate_required_columns,
)

from utils_ml import (
    compute_knn_ad,
    convert_ec3_to_y,
    convert_y_to_ec3,
    extract_invitro_calls,
    valid_regression_metrics,
)


TARGET_OUTPUT_COL = (
    "LLNA EC3(NS converted to 150%)"
)

META_COLUMNS = (
    "No.",
    "Name",
    "CAS Number",
    "SMILES",
)


def load_artifact(
    artifact_path: str | os.PathLike,
) -> dict:
    """
    artifactを読み込み、
    推論に必要なキーを検査する。
    """

    artifact = joblib.load(
        artifact_path
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "artifact は dict "
            "である必要があります。"
        )

    required = [
        "full_feature_columns",
        "selected_feature_mask",
        "selected_feature_columns",
        "folds",
    ]

    missing = [
        k
        for k in required
        if k not in artifact
    ]

    if missing:
        raise KeyError(
            "artifact に必要キーが"
            f"ありません: {missing}"
        )

    full_cols = list(
        artifact[
            "full_feature_columns"
        ]
    )

    mask = np.asarray(
        artifact[
            "selected_feature_mask"
        ],
        dtype=bool,
    ).reshape(-1)

    selected_cols = list(
        artifact[
            "selected_feature_columns"
        ]
    )

    if len(full_cols) != len(mask):
        raise ValueError(
            "full_feature_columns と "
            "selected_feature_mask の"
            "長さが一致しません。"
        )

    if (
        int(mask.sum())
        != len(selected_cols)
    ):
        raise ValueError(
            "selected_feature_mask と "
            "selected_feature_columns が"
            "不整合です。"
        )

    if len(
        _get_folds(
            artifact
        )
    ) == 0:
        raise ValueError(
            "artifact にfoldがありません。"
        )

    return artifact


def _get_folds(
    artifact: dict,
) -> list[dict]:
    """
    foldsがlistでもdictでも
    fold_name付きlistとして返す。
    """

    raw = artifact[
        "folds"
    ]

    out: list[dict] = []

    if isinstance(
        raw,
        dict,
    ):

        for key, value in raw.items():

            if not isinstance(
                value,
                dict,
            ):
                raise TypeError(
                    f"fold {key!r} "
                    "がdictではありません。"
                )

            fold = dict(
                value
            )

            fold.setdefault(
                "fold_name",
                str(key),
            )

            out.append(
                fold
            )

    else:

        for i, value in enumerate(
            list(raw),
            start=1,
        ):

            if not isinstance(
                value,
                dict,
            ):
                raise TypeError(
                    f"fold_{i} "
                    "がdictではありません。"
                )

            fold = dict(
                value
            )

            fold.setdefault(
                "fold_name",
                f"fold_{i}",
            )

            out.append(
                fold
            )

    names = [
        str(
            f["fold_name"]
        )
        for f in out
    ]

    if len(names) != len(
        set(names)
    ):
        raise ValueError(
            "fold_name が重複しています:"
            f" {names}"
        )

    return out


def _find_target_column(
    input_df: pd.DataFrame,
    artifact: dict,
) -> str | None:

    candidates = [
        artifact.get(
            "target"
        ),
        TARGET_COLUMN_DEFAULT,
        TARGET_OUTPUT_COL,
        "LLNA_EC3",
        "EC3",
    ]

    for col in candidates:

        if (
            col
            and col
            in input_df.columns
        ):
            return str(
                col
            )

    return None


def _as_1d_float(
    values,
    *,
    name: str,
    expected_len: int | None = None,
) -> np.ndarray:

    arr = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if (
        expected_len is not None
        and len(arr) != expected_len
    ):
        raise ValueError(
            f"{name} の行数が不正です: "
            f"expected={expected_len}, "
            f"actual={len(arr)}"
        )

    return arr


def _read_molecular_weight(
    input_df: pd.DataFrame,
) -> np.ndarray:
    """
    EC3計算用MolWeightを
    入力CSVから直接取得する。

    MolWeightがselected featureでなくても
    EC3逆変換には必須。
    """

    if MW_COL not in input_df.columns:

        raise KeyError(
            f"入力CSVに '{MW_COL}' "
            "列がありません。"
            "pred_yからEC3へ戻すために"
            "MolWeightが必須です。"
        )

    mw_series = pd.to_numeric(
        input_df[MW_COL],
        errors="coerce",
    )

    mw = mw_series.to_numpy(
        dtype=float
    )

    invalid = (
        ~np.isfinite(mw)
        | (mw <= 0)
    )

    if invalid.any():

        bad_rows = np.flatnonzero(
            invalid
        )[:10].tolist()

        bad_values = (
            input_df
            .iloc[bad_rows][MW_COL]
            .tolist()
        )

        raise ValueError(
            f"'{MW_COL}' に"
            "欠損・非数値・0以下があります。"
            f" 該当行: {bad_rows}, "
            f"値: {bad_values}"
        )

    return mw


def prepare_input_features(
    input_df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    """
    artifact学習時の
    full_feature_columnsと
    同じ列・順序を作る。

    selected特徴量は必須。

    selectedでないfull特徴量が
    入力CSVに無ければNaNとして
    学習済みimputerへ渡す。
    """

    selected_cols = list(
        artifact[
            "selected_feature_columns"
        ]
    )

    ok, missing = (
        validate_required_columns(
            input_df,
            selected_cols,
        )
    )

    if not ok:
        raise KeyError(
            "入力CSVに必要な選択特徴量が"
            f"不足しています: {missing}"
        )

    full_cols = list(
        artifact[
            "full_feature_columns"
        ]
    )

    data = {}

    for col in full_cols:

        if col in input_df.columns:

            data[col] = pd.to_numeric(
                input_df[col],
                errors="coerce",
            )

        else:

            data[col] = np.nan

    X_full = pd.DataFrame(
        data,
        index=input_df.index,
    )

    return X_full.loc[
        :,
        full_cols,
    ]


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

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
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def _select_min_ec3_fold(
    pred_y_fold: pd.DataFrame,
    pred_ec3_fold: pd.DataFrame,
    *,
    eligibility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    各行について
    Predict_LLNA EC3 が
    最も小さいfoldを採用する。

    pred_yとEC3は必ず
    同じfoldから取得する。
    """

    if pred_ec3_fold.empty:
        raise ValueError(
            "fold予測結果が空です。"
        )

    columns = list(
        pred_ec3_fold.columns
    )

    if (
        list(
            pred_y_fold.columns
        )
        != columns
    ):
        raise ValueError(
            "pred_yとpred_ec3の"
            "fold列が一致していません。"
        )

    if (
        len(pred_y_fold)
        != len(pred_ec3_fold)
    ):
        raise ValueError(
            "pred_yとpred_ec3の"
            "行数が一致していません。"
        )

    y_values = (
        pred_y_fold
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    ec3_values = (
        pred_ec3_fold
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    valid = (
        np.isfinite(
            y_values
        )
        &
        np.isfinite(
            ec3_values
        )
        &
        (
            ec3_values > 0
        )
    )

    if eligibility is not None:

        eligible = (
            eligibility
            .reindex(
                index=(
                    pred_ec3_fold.index
                ),
                columns=columns,
            )
            .fillna(False)
            .astype(bool)
            .to_numpy(dtype=bool)
        )

        valid &= eligible

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

    rows = np.arange(
        len(pred_ec3_fold)
    )

    selected_y = np.full(
        len(pred_ec3_fold),
        np.nan,
        dtype=float,
    )

    selected_ec3 = np.full(
        len(pred_ec3_fold),
        np.nan,
        dtype=float,
    )

    selected_fold = np.full(
        len(pred_ec3_fold),
        None,
        dtype=object,
    )

    selected_y[
        has_candidate
    ] = y_values[
        rows[has_candidate],
        chosen_idx[has_candidate],
    ]

    selected_ec3[
        has_candidate
    ] = ec3_values[
        rows[has_candidate],
        chosen_idx[has_candidate],
    ]

    fold_names = np.asarray(
        columns,
        dtype=object,
    )

    selected_fold[
        has_candidate
    ] = fold_names[
        chosen_idx[has_candidate]
    ]

    return pd.DataFrame(
        {
            "selected_fold":
                selected_fold,

            "selected_fold_index":
                np.where(
                    has_candidate,
                    chosen_idx,
                    -1,
                ),

            "pred_y":
                selected_y,

            "pred_ec3":
                selected_ec3,

            "has_candidate":
                has_candidate,
        },
        index=(
            pred_ec3_fold.index
        ),
    )


def _selected_bool_from_fold(
    bool_fold: pd.DataFrame,
    selection: pd.DataFrame,
) -> np.ndarray:

    values = (
        bool_fold
        .fillna(False)
        .astype(bool)
        .to_numpy(dtype=bool)
    )

    idx = selection[
        "selected_fold_index"
    ].to_numpy(
        dtype=int
    )

    has = selection[
        "has_candidate"
    ].to_numpy(
        dtype=bool
    )

    rows = np.arange(
        len(bool_fold)
    )

    result = np.zeros(
        len(bool_fold),
        dtype=bool,
    )

    result[
        has
    ] = values[
        rows[has],
        idx[has],
    ]

    return result


def _safe_invitro_output(
    X_full: pd.DataFrame,
    n_rows: int,
) -> pd.DataFrame:
    """
    in vitro列不足時でも
    EC3主予測を失敗させない。
    """

    try:

        invitro = (
            extract_invitro_calls(
                X_full
            )
            .reset_index(
                drop=True
            )
        )

        if (
            len(invitro)
            != n_rows
        ):
            raise ValueError(
                "in vitro結果の"
                "行数が一致しません。"
            )

        return invitro

    except Exception:

        return pd.DataFrame(
            {
                "invitro_AD":
                    pd.Series(
                        [pd.NA]
                        * n_rows,
                        dtype="boolean",
                    ),

                "vitro_call_sum":
                    np.full(
                        n_rows,
                        np.nan,
                    ),
            }
        )


def predict_with_artifact(
    artifact: dict,
    input_df: pd.DataFrame,
    *,
    compute_ad: bool = True,

    # 旧Streamlitとの互換用。
    # 指定されても実際には
    # min EC3方式を使う。
    aggregation: str | None = None,
) -> pd.DataFrame:
    """
    artifactで予測する。

    代表値は常に:
        各行で予測EC3が最小のfold

    pred_yもその同じfoldから取得する。
    """

    if not isinstance(
        input_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "input_df は"
            "pandas.DataFrame "
            "である必要があります。"
        )

    if input_df.empty:
        raise ValueError(
            "入力データが0行です。"
        )

    # ==================================================
    # MolWeight
    # ==================================================
    # EC3計算ではinput_dfから直接読む。
    # selected/full feature経由にしない。
    mw = _read_molecular_weight(
        input_df
    )

    # ==================================================
    # Model input
    # ==================================================
    X_full = prepare_input_features(
        input_df,
        artifact,
    )

    selected_mask_global = np.asarray(
        artifact[
            "selected_feature_mask"
        ],
        dtype=bool,
    ).reshape(-1)

    if (
        len(selected_mask_global)
        != X_full.shape[1]
    ):
        raise ValueError(
            "artifact mask長と"
            "full_feature_columns数が"
            "一致しません。"
        )

    pred_y_fold = pd.DataFrame(
        index=input_df.index
    )

    pred_ec3_fold = pd.DataFrame(
        index=input_df.index
    )

    ad_fold = pd.DataFrame(
        index=input_df.index
    )

    folds = _get_folds(
        artifact
    )

    ad_cfg = (
        artifact.get(
            "ad_config",
            {},
        )
        or {}
    )

    # ==================================================
    # 全foldで推論
    # ==================================================
    for i, fold in enumerate(
        folds,
        start=1,
    ):

        fold_name = str(
            fold.get(
                "fold_name",
                f"fold_{i}",
            )
        )

        if (
            "pipeline"
            not in fold
            or
            "model"
            not in fold
        ):
            raise KeyError(
                f"{fold_name} に "
                "pipeline/model が"
                "ありません。"
            )

        pipe = fold[
            "pipeline"
        ]

        model = fold[
            "model"
        ]

        # fold個別maskがあれば優先
        fold_mask = np.asarray(
            fold.get(
                "selected_feature_mask",
                selected_mask_global,
            ),
            dtype=bool,
        ).reshape(-1)

        if (
            len(fold_mask)
            != X_full.shape[1]
        ):
            raise ValueError(
                f"{fold_name}: "
                "selected_feature_mask"
                "の長さが不正です。"
            )

        # 学習済みpipelineをそのままtransform
        X_pre = pipe.transform(
            X_full
        )

        X_pre = np.asarray(
            X_pre,
            dtype=float,
        )

        if X_pre.ndim != 2:
            raise ValueError(
                f"{fold_name}: "
                "pipeline.transform出力が"
                "2次元ではありません。"
            )

        if (
            X_pre.shape[1]
            != len(fold_mask)
        ):
            raise ValueError(
                f"{fold_name}: "
                "前処理後列数と"
                "mask長が一致しません。"
            )

        X_sel = X_pre[
            :,
            fold_mask,
        ]

        expected_selected = len(
            fold.get(
                "selected_feature_columns",
                artifact[
                    "selected_feature_columns"
                ],
            )
        )

        if (
            X_sel.shape[1]
            != expected_selected
        ):
            raise ValueError(
                f"{fold_name}: "
                "選択後特徴量数が不正です。 "
                f"actual={X_sel.shape[1]}, "
                f"expected={expected_selected}"
            )

        # ==============================================
        # y予測
        # ==============================================
        pred_y = _as_1d_float(
            model.predict(
                X_sel
            ),
            name=(
                f"{fold_name}."
                "pred_y"
            ),
            expected_len=len(
                input_df
            ),
        )

        # ==============================================
        # y -> EC3
        # ==============================================
        pred_ec3 = _as_1d_float(
            convert_y_to_ec3(
                pred_y,
                mw,
            ),
            name=(
                f"{fold_name}."
                "pred_ec3"
            ),
            expected_len=len(
                input_df
            ),
        )

        pred_y_fold[
            fold_name
        ] = pred_y

        pred_ec3_fold[
            fold_name
        ] = pred_ec3

        # ==============================================
        # AD
        # ==============================================
        if compute_ad:

            if (
                "X_train_selected"
                not in fold
            ):
                raise KeyError(
                    f"{fold_name} に "
                    "X_train_selected "
                    "がありません。"
                )

            X_train_selected = (
                np.asarray(
                    fold[
                        "X_train_selected"
                    ],
                    dtype=float,
                )
            )

            if (
                X_train_selected.ndim
                != 2
                or
                X_train_selected.shape[1]
                != X_sel.shape[1]
            ):
                raise ValueError(
                    f"{fold_name}: "
                    "AD学習特徴量数と"
                    "入力特徴量数が"
                    "一致しません。"
                )

            k = int(
                fold.get(
                    "ad_knn_k",
                    ad_cfg.get(
                        "knn_k",
                        5,
                    ),
                )
            )

            coverage = float(
                fold.get(
                    "ad_coverage",
                    ad_cfg.get(
                        "coverage",
                        ad_cfg.get(
                            "AD_coverage",
                            0.1,
                        ),
                    ),
                )
            )

            (
                ad_bool,
                _,
                _,
            ) = compute_knn_ad(
                X_train_selected,
                X_sel,
                knn_k=k,
                coverage=coverage,
            )

            ad_bool = np.asarray(
                ad_bool,
                dtype=bool,
            ).reshape(-1)

            if (
                len(ad_bool)
                != len(input_df)
            ):
                raise ValueError(
                    f"{fold_name}: "
                    "AD結果の行数が"
                    "一致しません。"
                )

            ad_fold[
                fold_name
            ] = ad_bool

        else:

            ad_fold[
                fold_name
            ] = np.ones(
                len(input_df),
                dtype=bool,
            )

    # ==================================================
    # EC3存在チェック
    # ==================================================
    ec3_values = (
        pred_ec3_fold
        .to_numpy(dtype=float)
    )

    valid_ec3 = (
        np.isfinite(
            ec3_values
        )
        &
        (
            ec3_values > 0
        )
    )

    if not valid_ec3.any():

        raise ValueError(
            "全foldのPredict_LLNA EC3が"
            "無効です。"
            "MolWeight、convert_y_to_ec3、"
            "artifact/modelを確認してください。"
        )

    # ==================================================
    # 全foldのEC3最小
    # ==================================================
    selected = _select_min_ec3_fold(
        pred_y_fold,
        pred_ec3_fold,
    )

    # ==================================================
    # AD内foldだけのEC3最小
    # ==================================================
    selected_in_ad = _select_min_ec3_fold(
        pred_y_fold,
        pred_ec3_fold,
        eligibility=ad_fold,
    )

    # ==================================================
    # 出力作成
    # ==================================================
    out = pd.DataFrame(
        index=np.arange(
            len(input_df)
        )
    )

    meta_cols = [
        c
        for c in META_COLUMNS
        if c in input_df.columns
    ]

    if meta_cols:

        out = pd.concat(
            [
                input_df[
                    meta_cols
                ].reset_index(
                    drop=True
                ),
                out,
            ],
            axis=1,
        )

    # 実際にEC3計算に使ったMWも出す
    out[MW_COL] = mw

    # ==================================================
    # 真値があれば取得
    # ==================================================
    target_col = _find_target_column(
        input_df,
        artifact,
    )

    if target_col is not None:

        y_ec3 = pd.to_numeric(
            input_df[
                target_col
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid_true = (
            np.isfinite(
                y_ec3
            )
            &
            (
                y_ec3 > 0
            )
        )

        y_true = np.full(
            len(input_df),
            np.nan,
            dtype=float,
        )

        if valid_true.any():

            converted = _as_1d_float(
                convert_ec3_to_y(
                    y_ec3[
                        valid_true
                    ],
                    mw[
                        valid_true
                    ],
                ),
                name="true_y",
                expected_len=int(
                    valid_true.sum()
                ),
            )

            y_true[
                valid_true
            ] = converted

    else:

        y_ec3 = np.full(
            len(input_df),
            np.nan,
            dtype=float,
        )

        y_true = np.full(
            len(input_df),
            np.nan,
            dtype=float,
        )

    out[
        "true_y"
    ] = y_true

    out[
        TARGET_OUTPUT_COL
    ] = y_ec3

    # ==================================================
    # 代表値
    # 「EC3最小fold」
    # ==================================================
    out[
        "selected_fold"
    ] = (
        selected[
            "selected_fold"
        ]
        .reset_index(
            drop=True
        )
    )

    out[
        "pred_y"
    ] = (
        selected[
            "pred_y"
        ]
        .to_numpy(
            dtype=float
        )
    )

    out[
        "Predict_LLNA EC3"
    ] = (
        selected[
            "pred_ec3"
        ]
        .to_numpy(
            dtype=float
        )
    )

    out[
        "Fold_misprediction"
    ] = _safe_ratio(
        out[
            "Predict_LLNA EC3"
        ],
        out[
            TARGET_OUTPUT_COL
        ],
    )

    # ==================================================
    # AD
    # ==================================================
    out[
        "kNN_AD"
    ] = _selected_bool_from_fold(
        ad_fold,
        selected,
    )

    out[
        "any_kNN_AD"
    ] = (
        ad_fold
        .fillna(False)
        .astype(bool)
        .any(
            axis=1
        )
        .to_numpy()
    )

    # AD内foldだけでEC3最小
    out[
        "knn_inAD_selected_fold"
    ] = (
        selected_in_ad[
            "selected_fold"
        ]
        .reset_index(
            drop=True
        )
    )

    out[
        "knn_inAD_pred_y"
    ] = (
        selected_in_ad[
            "pred_y"
        ]
        .to_numpy(
            dtype=float
        )
    )

    out[
        "knn_inAD_Predict_LLNA EC3"
    ] = (
        selected_in_ad[
            "pred_ec3"
        ]
        .to_numpy(
            dtype=float
        )
    )

    out[
        "knn_inAD_Fold_misprediction"
    ] = _safe_ratio(
        out[
            "knn_inAD_Predict_LLNA EC3"
        ],
        out[
            TARGET_OUTPUT_COL
        ],
    )

    # ==================================================
    # in vitro
    # ==================================================
    invitro_df = _safe_invitro_output(
        X_full,
        len(input_df),
    )

    out[
        "invitro_AD"
    ] = (
        invitro_df[
            "invitro_AD"
        ]
        .to_numpy()
    )

    out[
        "vitro_call_sum"
    ] = pd.to_numeric(
        invitro_df[
            "vitro_call_sum"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    # ==================================================
    # fold別結果
    # ==================================================
    out = pd.concat(
        [
            out.reset_index(
                drop=True
            ),

            ad_fold
            .add_prefix(
                "knn_AD_"
            )
            .reset_index(
                drop=True
            ),

            pred_y_fold
            .add_prefix(
                "pred_y_"
            )
            .reset_index(
                drop=True
            ),

            pred_ec3_fold
            .add_prefix(
                "Predict_LLNA EC3_"
            )
            .reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # ==================================================
    # 統計列
    # ==================================================
    out[
        "pred_y_mean"
    ] = (
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

    out[
        "pred_y_max"
    ] = (
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

    # ==================================================
    # 検算
    # 代表EC3とfold最小EC3は
    # 必ず一致するはず
    # ==================================================
    representative = out[
        "Predict_LLNA EC3"
    ].to_numpy(
        dtype=float
    )

    calculated_min = out[
        "Predict_LLNA EC3_min"
    ].to_numpy(
        dtype=float
    )

    comparable = (
        np.isfinite(
            representative
        )
        &
        np.isfinite(
            calculated_min
        )
    )

    if (
        comparable.any()
        and
        not np.allclose(
            representative[
                comparable
            ],
            calculated_min[
                comparable
            ],
            rtol=1e-10,
            atol=1e-12,
        )
    ):

        raise RuntimeError(
            "代表Predict_LLNA EC3と"
            "fold最小EC3が一致しません。"
        )

    # ==================================================
    # metrics
    # ==================================================
    metrics = valid_regression_metrics(
        y_true,
        out[
            "pred_y"
        ].to_numpy(
            dtype=float
        ),
    )

    out.attrs[
        "metrics"
    ] = metrics

    out.attrs[
        "aggregation"
    ] = (
        "minimum_predicted_ec3_per_row"
    )

    out.attrs[
        "fold_count"
    ] = len(
        folds
    )

    return out


def _make_prediction_template(
    selected_feature_columns: Sequence[str],
    output_path: str | os.PathLike,
) -> Path:
    """
    MolWeightを必ず含む
    予測テンプレートを作る。
    """

    columns: list[str] = []

    for col in [
        *META_COLUMNS,
        MW_COL,
        *selected_feature_columns,
    ]:

        if col not in columns:
            columns.append(
                col
            )

    path = Path(
        output_path
    )

    ensure_dir(
        path.parent
    )

    pd.DataFrame(
        columns=columns
    ).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(
        description=(
            "artifactベース予測ランタイム "
            "（EC3最小fold採用）"
        )
    )

    p.add_argument(
        "--artifact-path",
        required=True,
    )

    p.add_argument(
        "--input-csv",
        required=False,
    )

    p.add_argument(
        "--output-csv",
        default="prediction_results.csv",
    )

    p.add_argument(
        "--template-csv",
        default=None,
    )

    p.add_argument(
        "--no-ad",
        action="store_true",
    )

    return p.parse_args()


def main() -> None:

    args = parse_args()

    artifact = load_artifact(
        args.artifact_path
    )

    # ==================================================
    # Template
    # ==================================================
    if args.template_csv:

        template_path = (
            _make_prediction_template(
                artifact[
                    "selected_feature_columns"
                ],
                args.template_csv,
            )
        )

        print(
            "[DONE] template saved:",
            template_path,
        )

        print(
            "[INFO] template includes:",
            MW_COL,
        )

    # ==================================================
    # Prediction
    # ==================================================
    if args.input_csv:

        df, enc = read_csv_auto(
            args.input_csv
        )

        print(
            "[INFO] input encoding:",
            enc,
        )

        result = predict_with_artifact(
            artifact,
            df,
            compute_ad=(
                not args.no_ad
            ),
        )

        output_csv = Path(
            args.output_csv
        )

        ensure_dir(
            output_csv.parent
        )

        metrics = dict(
            result.attrs.get(
                "metrics",
                {},
            )
        )

        result_to_save = (
            result.copy()
        )

        # Streamlit/CSV互換用
        result_to_save.attrs = {}

        result_to_save = (
            result_to_save.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

        result_to_save.to_csv(
            output_csv,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "[DONE] prediction saved:",
            output_csv,
        )

        print(
            "[INFO] aggregation:"
            " minimum predicted EC3 "
            "fold per row"
        )

        print(
            "[INFO] Predict_LLNA EC3 "
            "non-null:",
            int(
                result_to_save[
                    "Predict_LLNA EC3"
                ]
                .notna()
                .sum()
            ),
        )

        print(
            "[METRICS]",
            metrics,
        )


if __name__ == "__main__":
    main()
