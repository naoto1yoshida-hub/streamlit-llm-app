import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 環境変数ロード
load_dotenv()

# APIキー確認
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI APIキーが設定されていません。環境変数を確認してください。")
    st.stop()

# LLM初期化（最新版は "model"）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LLMからの回答を取得する関数
def get_llm_response(input_text, expert_type):

    if expert_type == "料理の専門家":
        system_message = (
            "あなたはプロの料理専門家です。\n"
            "家庭料理から本格的な専門料理まで幅広く精通し、ユーザーの目的・食材・調理レベルに合わせて最適なレシピを提案します。\n"
            "- 材料、分量、手順を具体的に示す\n"
            "- 初心者にも再現しやすく説明\n"
            "- 代替案や失敗しにくいコツを含める\n"
            "- 条件（時間・予算・器具）を考慮\n"
            "- 安全性に配慮"
        )
    elif expert_type == "旅行の専門家":
        system_message = (
            "あなたはプロの旅行プランナーです。\n"
            "ユーザーの希望をヒアリングし、実用的な旅行プランを作成します。\n"
            "- 日程、交通、所要時間、費用を具体的に\n"
            "- 動線に無理がないプラン\n"
            "- 代替案の提示\n"
            "- 過剰提案を避ける\n"
            "- 専門家として明確に回答"
        )
    else:
        system_message = "あなたは一般的な知識を持つアシスタントです。"

    # メッセージ生成
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=input_text)
    ]

    # 最新版は invoke を使う
    response = llm.invoke(messages)
    return response.content

# ======================
# Streamlit UI
# ======================

st.title("LLMを使った専門家チャットアプリ")
# ▼▼ 追加した説明テキスト ▼▼
st.markdown("""
### 🔍 アプリ概要
このアプリは、LLM（大規模言語モデル）を活用して **料理** または **旅行** の専門家として回答を返すチャットアプリです。  
質問内容に応じて、専門知識をもった AI が最適なアドバイスを提供します。

---

### 📝 操作方法
1. **専門家の種類（料理 または 旅行）を選択**  
2. **質問内容を入力**  
3. **「送信」ボタンを押すと、専門家としての回答が表示されます**

---

必要な情報が足りない場合は、AI が追加の質問を行うことがあります。
""")
# ▲▲ ここまで説明文 ▲▲

expert_type = st.radio(
    "専門家の種類を選択してください：",
    ("料理の専門家", "旅行の専門家")
)

input_text = st.text_input("質問を入力してください：")

if st.button("送信"):
    if input_text.strip():
        response = get_llm_response(input_text, expert_type)
        st.subheader("LLMからの回答")
        st.write(response)
    else:
        st.error("質問を入力してください。")
