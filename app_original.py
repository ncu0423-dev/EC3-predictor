from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from original_runtime import (
    CANONICAL_FEATURE_COLUMNS,
    MW_COL,
    load_artifact,
    predict_with_artifact,
)

ARTIFACT_PATH = Path("artifacts/original_model_artifact.joblib")


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """UTF-8 / CP932 を順に試してCSVを読む。"""
    raw = uploaded_file.getvalue()
    last_error: Exception | None = None

    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(BytesIO(raw), encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_error = e

    raise ValueError(f"CSVを読み込めませんでした。最後のエラー: {last_error}")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return output.getvalue()


st.set_page_config(
    page_title="Original XGBoost Inference App",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Original XGBoost Inference App")
st.caption(
    "元Notebookの学習済みアーティファクトをそのまま読み込み、再学習せずに推論します。"
)

if not ARTIFACT_PATH.exists():
    st.error(
        "アーティファクトが見つかりません。まず Notebook から "
        "`save_original_artifact.py` を使って "
        "`artifacts/original_model_artifact.joblib` を作成してください。"
    )
    st.stop()

artifact = load_artifact(ARTIFACT_PATH)

with st.expander("必要な入力列", expanded=False):
    st.markdown("### 最小入力")
    st.write([MW_COL] + CANONICAL_FEATURE_COLUMNS)
    st.caption(
        "この最小入力モードは、21特徴量に欠損がない場合に限って、"
        "選択列部分の推論を元Notebookに一致させます。"
    )
    st.markdown("### 厳密モード")
    st.write(
        "学習時の全特徴量列（artifact['full_feature_columns']）を含むCSVを入力すると、"
        "foldごとの Pipeline.transform をそのまま使う厳密モードになります。"
    )

uploaded = st.file_uploader("予測用CSVをアップロードしてください。", type=["csv"])

compute_ad = st.checkbox("AD列も計算する", value=True)

if uploaded is not None:
    try:
        input_df = read_uploaded_csv(uploaded)

        st.subheader("入力データ")
        st.dataframe(input_df, use_container_width=True)

        result_df = predict_with_artifact(
            artifact=artifact,
            input_df=input_df,
            compute_ad=compute_ad,
        )

        st.subheader("予測結果")
        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            label="予測結果CSVをダウンロード",
            data=dataframe_to_csv_bytes(result_df),
            file_name="prediction_results_original.csv",
            mime="text/csv",
        )

    except Exception as e:  # noqa: BLE001
        st.error(str(e))
