# Streamlit Community Cloud (Linux) ChromaDB SQLite3 compatibility fix
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine, DOMAINS

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Google Antigravity RAG 챗봇 & 6시간 마스터 대시보드",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -1em;
    }
    .citation-badge {
        display: inline-block;
        background-color: #e8f0fe;
        color: #1a73e8;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #d2e3fc;
    }
    .score-badge {
        background-color: #e6f4ea;
        color: #137333;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .curriculum-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1a73e8;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ RAG 스튜디오 설정")
    
    # 1. API Key Resolution (Secrets -> Env -> Manual Input)
    secrets_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            secrets_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    env_api_key = secrets_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    
    # 0. Demo Mode Toggle
    demo_mode = st.toggle("🧪 체험(데모) 모드로 실행", value=(not bool(env_api_key)), help="API 키 없이도 사전 정의된 4대 분야 모범 답변과 RAG 동작을 즉시 체험할 수 있습니다.")

    if not demo_mode:
        api_key = st.text_input(
            "Gemini API Key",
            value=env_api_key,
            type="password",
            help="Google AI Studio에서 발급받은 API Key를 입력하세요."
        )
    else:
        api_key = ""
        st.info("💡 데모 모드 활성화됨: API 키 없이 동작합니다.")

    st.markdown("---")
    st.subheader("🎯 실습 분야 선택")
    domain_choice = st.selectbox(
        "도메인 시나리오",
        options=list(DOMAINS.keys()),
        format_func=lambda k: DOMAINS[k]["title"]
    )

    st.markdown("---")
    st.subheader("🔍 검색 튜닝 (Retrieval)")
    top_k = st.slider("Top-K (참조 청크 수)", min_value=1, max_value=5, value=3)
    score_threshold = st.slider("유사도 임계값 (Score Threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if st.button("🔄 지식 베이스 재색인 (Re-Index)"):
        with st.spinner("문서 청킹 및 ChromaDB 색인 중..."):
            engine_temp = RAGEngine(api_key=api_key, domain_key=domain_choice)
            count = engine_temp.build_knowledge_base(force_rebuild=True)
            st.success(f"{count}개 청크 색인 완료!")

# ----------------- MAIN TABS -----------------
tab1, tab2 = st.tabs(["💬 4대 분야 RAG 챗봇 실습", "📖 6시간 완성 교안 보기"])

# ================= TAB 1: RAG CHATBOT =================
with tab1:
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.subheader(DOMAINS[domain_choice]["title"])
    with col_t2:
        if st.button("💡 추천 테스트 질문 입력"):
            st.session_state["sample_prompt"] = DOMAINS[domain_choice]["demo_q"]

    # Session State for Chat History
    if f"messages_{domain_choice}" not in st.session_state:
        st.session_state[f"messages_{domain_choice}"] = []

    # Display History
    for msg in st.session_state[f"messages_{domain_choice}"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander(f"📚 참조 출처 ({len(msg['citations'])}개 청크)"):
                    for c in msg["citations"]:
                        src = c.get("metadata", {}).get("source", "문서")
                        score = c.get("score", 0.0)
                        st.markdown(f"<span class='citation-badge'>{src}</span> <span class='score-badge'>유사도: {score}</span>", unsafe_allow_html=True)
                        st.caption(c.get("text", ""))

    # Input Box
    default_input = st.session_state.pop("sample_prompt", "")
    user_input = st.chat_input("질문을 입력하세요... (우측 상단 '추천 질문' 클릭 가능)")

    query_to_send = user_input or default_input

    if query_to_send:
        # User message
        st.session_state[f"messages_{domain_choice}"].append({"role": "user", "content": query_to_send})
        with st.chat_message("user"):
            st.markdown(query_to_send)

        # Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("지식 베이스 검색 및 응답 생성 중..."):
                engine = RAGEngine(api_key=api_key, domain_key=domain_choice)
                res = engine.generate_response(
                    query=query_to_send,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    demo_mode=demo_mode
                )
                st.markdown(res["answer"])
                
                if res["citations"]:
                    with st.expander(f"📚 참조 출처 ({len(res['citations'])}개 청크)"):
                        for c in res["citations"]:
                            src = c.get("metadata", {}).get("source", "문서")
                            score = c.get("score", 0.0)
                            st.markdown(f"<span class='citation-badge'>{src}</span> <span class='score-badge'>유사도: {score}</span>", unsafe_allow_html=True)
                            st.caption(c.get("text", ""))

                st.session_state[f"messages_{domain_choice}"].append({
                    "role": "assistant",
                    "content": res["answer"],
                    "citations": res["citations"]
                })

# ================= TAB 2: CURRICULUM VIEWER =================
with tab2:
    st.subheader("📚 Google Antigravity 기반 실무 RAG 챗봇 구축 6시간 마스터 교안")
    curriculum_path = "Google Antigravity RAG 챗봇 강의 교안.md"
    
    if os.path.exists(curriculum_path):
        with open(curriculum_path, "r", encoding="utf-8") as f:
            curriculum_text = f.read()
        
        st.download_button(
            label="📥 강의 교안 마크다운(.md) 다운로드",
            data=curriculum_text,
            file_name="Google_Antigravity_RAG_챗봇_강의_교안.md",
            mime="text/markdown"
        )
        st.markdown("---")
        st.markdown(curriculum_text)
    else:
        st.warning("강의 교안 파일을 찾을 수 없습니다.")
