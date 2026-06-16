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
    # すべてのエンコードで失敗した場合にエラー
    raise ValueError(f"CSVファイルを読み込めませんでした（エンコードエラー）。最後のエラー: {last_error}")

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return output.getvalue()

# ページ設定
st.set_page_config(
    page_title="Original XGBoost Inference App",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Original XGBoost Inference App")
st.caption("元Notebookで学習したXGBoostモデルを利用してEC3を予測します。")

# モデルアーティファクトの存在チェック
if not ARTIFACT_PATH.exists():
    st.error(
        "学習済みモデルが見つかりません。\n"
        "まず Notebook から `save_original_artifact.py` を使って "
        "`artifacts/original_model_artifact.joblib` を作成してください。"
    )
    st.stop()

artifact = load_artifact(ARTIFACT_PATH)
required_columns = [MW_COL] + CANONICAL_FEATURE_COLUMNS

st.markdown("---")

# テンプレートCSVダウンロードボタン
st.subheader("📥 入力テンプレートCSV")
template_df = pd.DataFrame(columns=required_columns)
st.download_button(
    label="テンプレートCSVをダウンロード",
    data=dataframe_to_csv_bytes(template_df),
    file_name="template.csv",
    mime="text/csv"
)

st.markdown("---")

# 必要な入力列の表示
with st.expander("必要な入力列", expanded=False):
    st.write(required_columns)

st.markdown("---")

# 入力例の表示
with st.expander("入力例（最小限のダミーデータ）", expanded=False):
    example_df = pd.DataFrame([{col: 0 for col in required_columns}])
    st.dataframe(example_df, width="stretch")

st.markdown("---")

# CSVアップロード
uploaded = st.file_uploader("予測用CSVをアップロードしてください", type=["csv"])
compute_ad = st.checkbox("Applicability Domain (AD) を計算する", value=True)

if uploaded is not None:
    try:
        input_df = read_uploaded_csv(uploaded)
        st.subheader("入力データ")
        st.dataframe(input_df.head(), width="stretch")

        # 必須列チェック
        missing_columns = [col for col in required_columns if col not in input_df.columns]
        if missing_columns:
            st.error("以下の必須列が不足しています:")
            st.write(missing_columns)
            st.stop()

        # 欠損値チェック
        na_counts = input_df[required_columns].isnull().sum()
        na_counts = na_counts[na_counts > 0]
        if not na_counts.empty:
            st.warning("必須列に欠損値があります（実際の値の予測精度に影響する可能性があります）:")
            st.dataframe(na_counts.rename("欠損数"), width="stretch")
        st.write("STEP 1: upload detected")

        input_df = read_uploaded_csv(uploaded)

        st.write("STEP 2: csv loaded")
        st.write(input_df.shape)

        missing_columns = [
            col for col in required_columns
            if col not in input_df.columns
        ]

        st.write("STEP 3: column check finished")

        st.write("STEP 4: prediction start")

        result_df = predict_with_artifact(
            artifact=artifact,
            input_df=input_df,
            compute_ad=False,
        )

        st.write("STEP 5: prediction finished")
        st.write(result_df.shape)

        st.success("予測が完了しました")
        # 予測結果の表示・ダウンロード
        st.subheader("予測結果")

        st.write(
            f"Rows: {result_df.shape[0]}, Columns: {result_df.shape[1]}"
        )

        display_cols = [
            col for col in result_df.columns
            if "Predict" in col
            or "pred" in col.lower()
        ]

        if len(display_cols) > 0:
            st.dataframe(
                result_df[display_cols],
                width="stretch"
            )
        else:
            st.dataframe(
                result_df.iloc[:, :20],
                width="stretch"
            )

        st.download_button(
            label="予測結果CSVをダウンロード",
            data=dataframe_to_csv_bytes(result_df),
            file_name="prediction_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(
            "予測中にエラーが発生しました（以下のトレースバックを参照してください）"
        )
        st.exception(e)