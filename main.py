import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import pandas as pd
import numpy as np
import random
import faker
import os

fake = faker.Faker("ja_JP")

# 業種リスト
industries = ["IT", "製造", "小売", "物流", "教育", "医療", "金融", "不動産", "広告", "飲食"]

# 都道府県
prefectures = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県",
    "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県",
    "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
]

# データ数
n = 100

# データ生成
data = []
for i in range(n):
    company = fake.company()
    industry = random.choice(industries)
    employee_count = np.random.randint(5, 5001)
    contact_name = fake.name()
    email = fake.email()
    location = random.choice(prefectures)
    budget = np.random.randint(1_000_000, 100_000_001)
    contact_count = np.random.randint(0, 11)
    score = np.random.randint(0, 101)

    data.append({
        "顧客ID": f"LEAD_{i+1:04d}",
        "会社名": company,
        "業種": industry,
        "従業員数": employee_count,
        "担当者名": contact_name,
        "メールアドレス": email,
        "都道府県": location,
        "年間予算（円）": budget,
        "過去の接触回数": contact_count,
        "成約確度スコア": score
    })

df = pd.DataFrame(data)

st.subheader("営業専用データ")
st.dataframe(df)

#クラスタごとの平均値
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# データフレーム（前ステップで生成済みの df を使用）
# 数値項目のみ抽出
features = ["従業員数", "年間予算（円）", "過去の接触回数", "成約確度スコア"]
X = df[features]

# 標準化（スケーリング）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# クラスタ数の設定（例：5）
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
df["クラスタ"] = kmeans.fit_predict(X_scaled)

# クラスタごとの平均値を表示
cluster_summary = df.groupby("クラスタ")[features].mean().round(1)
st.subheader("クラスターごとの平均値")
st.dataframe(cluster_summary)

import japanize_matplotlib
# PCAで次元削減（2Dプロット用）
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# クラスタごとの分布を可視化

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="クラスタ", palette="tab10", ax=ax)
ax.set_title("営業顧客クラスタリング（PCA可視化）")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend(title="クラスタ")
ax.grid(True)
st.subheader("K-means法による分類")
# Streamlitに表示
st.pyplot(fig)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 特徴量の抽出
features = ["従業員数", "年間予算（円）", "過去の接触回数", "成約確度スコア"]
X = df[features]

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA実行（2次元）
pca = PCA(n_components=2)
pca.fit(X_scaled)

# 因子負荷量の取得（PCA.components_）
loadings = pd.DataFrame(
    pca.components_.T,  # 転置して変数×主成分の形に
    columns=["主成分1", "主成分2"],
    index=features
)

# ヒートマップ描画
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
ax.set_title("主成分分析の因子負荷量（営業データ）")
ax.set_xlabel("主成分")
ax.set_ylabel("変数")
plt.tight_layout()
st.subheader("主成分分析を用いて描画したヒートマップ")
st.pyplot(fig)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm

llm =  ChatOpenAI(
    model="openai/gpt-oss-20b",
    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
    openai_api_base=st.secrets["OPENROUTER_API_BASE"],
    temperature=0.7
)

# プロンプトテンプレート
template = """
あなたは営業戦略の専門家です。
下記の情報に基づいて、今後の営業方針について具体的なアドバイスをしてください。ただし日本語で100字以内で回答してください。

会社名：{会社名}
過去の接触回数:{過去の接触回数}
成約確度スコア:{成約確度スコア}

"""

prompt = PromptTemplate(
    input_variables=["会社名","過去の接触回数","成約確度スコア"],
    template=template
)

# LLMチェーン構築
sales_advice_chain = LLMChain(llm=llm, prompt=prompt)

advice_result = []

# tqdmで進捗バー表示
for row in tqdm(df.itertuples(index=False), total=len(df), desc="営業アドバイス生成中"):
    input_text = {
        "会社名": row.会社名,
        "過去の接触回数": row.過去の接触回数,
        "成約確度スコア": row.成約確度スコア
    }

    advice = sales_advice_chain.run(input_text)

    print(f"\n✅ {row.会社名} のアドバイス生成完了 → {advice}\n")

    advice_result.append({
        "会社名": row.会社名,
        "営業アドバイス": advice
    })

for advice in advice_result:
    print(advice)

advice_df = pd.DataFrame(advice_result)
st.subheader("営業アドバイス")
st.dataframe(advice_df)