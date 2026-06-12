from __future__ import annotations

from io import BytesIO
from pathlib import Path
import traceback

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
    """UTF-8 / CP932 / Shift-JIS を順に試してCSVを読む"""
    raw = uploaded_file.getvalue()
    last_error = None

    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(BytesIO(raw), encoding=enc)
        except Exception as e:
            last_error = e

    raise ValueError(
        f"CSVを読み込めませんでした。\n\n最後のエラー:\n{last_error}"
    )


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
    "元Notebookで学習したXGBoostモデルを利用してEC3を予測します。"
)

if not ARTIFACT_PATH.exists():
    st.error(
        "学習済みモデルが見つかりません。\n\n"
        "artifacts/original_model_artifact.joblib を配置してください。"
    )
    st.stop()

artifact = load_artifact(ARTIFACT_PATH)

required_columns = [MW_COL] + CANONICAL_FEATURE_COLUMNS

st.markdown("---")

st.subheader("📥 テンプレートCSV")

template_df = pd.DataFrame(columns=required_columns)

st.download_button(
    label="テンプレートCSVをダウンロード",
    data=dataframe_to_csv_bytes(template_df),
    file_name="template.csv",
    mime="text/csv",
)

st.markdown("---")

with st.expander("必要な入力列", expanded=False):
    st.write(required_columns)

st.markdown("---")

with st.expander("入力例", expanded=False):

    example_df = pd.DataFrame(
        [
            {
                col: 0
                for col in required_columns
            }
        ]
    )

    st.dataframe(
        example_df,
        use_container_width=True,
    )

st.markdown("---")

uploaded = st.file_uploader(
    "予測用CSVをアップロードしてください",
    type=["csv"]
)

compute_ad = st.checkbox(
    "Applicability Domain (AD) を計算する",
    value=True
)

if uploaded is not None:

    try:

        input_df = read_uploaded_csv(uploaded)

        st.subheader("入力データ")

        st.dataframe(
            input_df.head(),
            use_container_width=True,
        )

        missing_columns = [
            col
            for col in required_columns
            if col not in input_df.columns
        ]

        if len(missing_columns) > 0:

            st.error(
                "以下の必須列が不足しています。"
            )

            st.write(missing_columns)

            st.stop()

        na_counts = input_df[required_columns].isnull().sum()

        na_counts = na_counts[na_counts > 0]

        if len(na_counts) > 0:

            st.warning(
                "必須列に欠損値があります。"
            )

            st.dataframe(
                na_counts.rename("欠損数"),
                use_container_width=True,
            )

        with st.spinner("予測実行中..."):

            result_df = predict_with_artifact(
                artifact=artifact,
                input_df=input_df,
                compute_ad=compute_ad,
            )

        st.success("予測が完了しました")

        st.subheader("予測結果")

        st.dataframe(
            result_df,
            use_container_width=True,
        )

        st.download_button(
            label="予測結果CSVをダウンロード",
            data=dataframe_to_csv_bytes(result_df),
            file_name="prediction_results.csv",
            mime="text/csv",
        )

    except Exception:

        st.error(
            "予測中にエラーが発生しました。"
        )

        st.code(
            traceback.format_exc(),
            language="python",
        )