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
    page_title="가람시 주차 조례 및 공공 RAG 마스터 포털",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# 🎨 [모바일 퍼스트 반응형 프리미엄 CSS]
# =============================================================================
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    * {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
    }
    
    /* 상단 헤더 안전 여백 확보: Streamlit 고정 상단바(Share, Deploy 바 등 약 60px) 아래로 온전히 노출되도록 패딩 조정 */
    .block-container {
        padding-top: 5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        max-width: 960px !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 4.4rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
    }

    /* Streamlit 상단 고정 헤더(Share, Deploy 바)와 탭 겹침 완벽 방지 */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(12px) !important;
        z-index: 99 !important;
        border-bottom: 1px solid rgba(226, 232, 240, 0.8) !important;
    }

    /* 탭 네비게이션 프리미엄 세그먼트 컨트롤 디자인 */
    div[data-testid="stTabs"] {
        margin-top: 0.35rem !important;
        margin-bottom: 1.25rem !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #f1f5f9 !important;
        padding: 6px !important;
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        display: flex !important;
        width: 100% !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab"] {
        flex: 1 1 0 !important;
        text-align: center !important;
        justify-content: center !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #475569 !important;
        background-color: transparent !important;
        border: none !important;
        white-space: nowrap !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div[data-testid="stTabs"] [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
    }

    /* Streamlit 기본 하단 밑줄 인디케이터 제거 (모던 필 스타일) */
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    /* 상단 공식 행정기관 배너 헤더 */
    .gov-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-radius: 16px;
        padding: 22px 24px;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(15, 52, 96, 0.25);
        margin-bottom: 18px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .gov-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        color: #64ffda;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    .gov-title {
        font-size: clamp(1.2rem, 3.5vw, 1.55rem);
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        line-height: 1.35;
    }
    .gov-subtitle {
        font-size: 0.88rem;
        color: #cbd5e1;
        margin-top: 6px;
        line-height: 1.45;
    }
    
    /* 4대 민원 인포 카드 */
    .info-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 14px 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 10px;
        color: #1e293b;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
        border-color: #3b82f6;
    }
    .info-card h4 {
        margin: 0 0 6px 0;
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f3460;
    }
    .info-card p {
        font-size: 0.82rem;
        color: #475569;
        margin: 0;
        line-height: 1.45;
    }
    .info-tag {
        display: inline-block;
        margin-top: 8px;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 4px;
        background: #f1f5f9;
        color: #2563eb;
        font-weight: 600;
    }
    
    /* 인용 배지 */
    .citation-badge {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82em;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #bae6fd;
    }
    .score-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.78em;
        font-weight: bold;
    }
    .curriculum-box {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        border-left: 5px solid #2563eb;
        margin-bottom: 15px;
        color: #334155;
    }
    .call-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #2563eb;
        color: white !important;
        padding: 8px 14px;
        border-radius: 20px;
        text-decoration: none;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 8px;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ RAG 스튜디오 설정")
    
    # 1. API Key Resolution
    secrets_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            secrets_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    env_api_key = secrets_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    
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
        format_func=lambda k: DOMAINS[k]["title"],
        index=0
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
    current_title = DOMAINS[domain_choice]["title"]
    
    # 모바일 공식 헤더 배너
    st.markdown(f"""
    <div class="gov-header">
        <span class="gov-badge">🏛️ 실무 특화 RAG AI 어시스턴트</span>
        <h1 class="gov-title">{current_title}</h1>
        <p class="gov-subtitle">
            스마트폰에서도 한눈에 파악되는 정보 중심 대시보드와 자치조례/실무규정 기반의 사실성 100% 원스톱 민원 상담을 제공합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 도메인별 4대 핵심 인포그래픽 대시보드 카드
    if domain_choice == "parking":
        st.markdown("##### 📌 시민들이 가장 많이 찾는 4대 주차 민원")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4>🚗 주차요금 & 감면</h4>
                <p>최초 30분 600원부터<br>경차·전기차 <b>50%</b>, 장애인 <b>80%</b> 감면</p>
                <span class="info-tag">가람시 조례 제4조</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card">
                <h4>📋 과태료 이의신청</h4>
                <p>사전납부 시 <b>20% 감경</b><br>부득이한 사유(고장/응급) 60일 내 접수</p>
                <span class="info-tag">질서위반법 제16조</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4>🚨 불법주정차 단속</h4>
                <p>일반도로 <b>10분 유예</b><br>6대 금지구역·어린이보호구역 <b>1분 단속</b></p>
                <span class="info-tag">도로교통법 제32조</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card">
                <h4>🅿️ 거주자우선주차</h4>
                <p>월 3~4만원대 이용<br>거주기간·친환경차·유공자 가점제</p>
                <span class="info-tag">가람시 조례 제12조</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 모바일 원터치 추천 질문 칩 (Quick Chips)
    st.markdown("##### 💬 터치 한 번으로 빠르게 질문하기")
    chip_cols = st.columns(4)
    preset_query = None

    if domain_choice == "parking":
        with chip_cols[0]:
            if st.button("🚨 단속 유예시간 몇 분?", use_container_width=True):
                preset_query = "가람시 불법주정차 단속 유예시간은 몇 분인가요? 어린이보호구역은요?"
        with chip_cols[1]:
            if st.button("🚗 공영주차장 감면 조건", use_container_width=True):
                preset_query = "공영주차장 주차요금과 경차, 전기차 할인 감면 혜택 알려줘요."
        with chip_cols[2]:
            if st.button("📋 병원 응급실 과태료 면제", use_container_width=True):
                preset_query = "응급환자 수송 때문에 주차위반 과태료가 나왔는데 이의신청 되나요?"
        with chip_cols[3]:
            if st.button("🅿️ 거주자우선주차 배점", use_container_width=True):
                preset_query = "거주자우선주차 신청 자격과 배점 기준, 이용 요금이 어떻게 되나요?"
    else:
        with chip_cols[0]:
            if st.button("💡 추천 테스트 질문 입력", use_container_width=True):
                preset_query = DOMAINS[domain_choice]["demo_q"]

    st.markdown("---")

    # Session State for Chat History
    if f"messages_{domain_choice}" not in st.session_state:
        st.session_state[f"messages_{domain_choice}"] = []

    # Display History
    for msg in st.session_state[f"messages_{domain_choice}"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if "citations" in msg and msg["citations"]:
                with st.expander(f"📚 법적 조례 및 근거 출처 ({len(msg['citations'])}개 청크)"):
                    for c in msg["citations"]:
                        src = c.get("metadata", {}).get("source", "문서")
                        score = c.get("score", 0.0)
                        st.markdown(f"<span class='citation-badge'>{src}</span> <span class='score-badge'>유사도: {score}</span>", unsafe_allow_html=True)
                        st.caption(c.get("text", ""))

    # Input Box
    user_input = st.chat_input("가람시 주차 조례 및 민원에 대해 질문하세요... (예: 점심시간 단속 유예되나요?)")
    query_to_send = preset_query or user_input

    if query_to_send:
        st.session_state[f"messages_{domain_choice}"].append({"role": "user", "content": query_to_send})
        with st.chat_message("user"):
            st.markdown(query_to_send)

        with st.chat_message("assistant"):
            with st.spinner("지식 베이스 검색 및 응답 생성 중..."):
                engine = RAGEngine(api_key=api_key, domain_key=domain_choice)
                res = engine.generate_response(
                    query=query_to_send,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    demo_mode=demo_mode
                )
                
                # 가람시 주차 도메인의 경우 스마트폰 직통 전화 버튼 추가
                answer_content = res["answer"]
                if domain_choice == "parking" and "052" not in answer_content:
                    answer_content += '\n\n<a href="tel:052120" class="call-btn">📞 가람시 교통민원콜센터(052-120) 바로 연결</a>'
                
                st.markdown(answer_content, unsafe_allow_html=True)
                
                if res["citations"]:
                    with st.expander(f"📚 법적 조례 및 근거 출처 ({len(res['citations'])}개 청크)"):
                        for c in res["citations"]:
                            src = c.get("metadata", {}).get("source", "문서")
                            score = c.get("score", 0.0)
                            st.markdown(f"<span class='citation-badge'>{src}</span> <span class='score-badge'>유사도: {score}</span>", unsafe_allow_html=True)
                            st.caption(c.get("text", ""))

                st.session_state[f"messages_{domain_choice}"].append({
                    "role": "assistant",
                    "content": answer_content,
                    "citations": res["citations"]
                })

# ================= TAB 2: CURRICULUM VIEWER =================
with tab2:
    st.markdown("""
    <div class="gov-header" style="background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);">
        <span class="gov-badge" style="color: #93c5fd;">📘 실무 RAG 마스터 과정</span>
        <h1 class="gov-title">Google Antigravity 기반 실무 RAG 챗봇 6시간 마스터 교안</h1>
        <p class="gov-subtitle">데이터 수집부터 조례 청킹, 하이브리드 검색, 모바일 UI/UX 배포까지 1일 완성 커리큘럼입니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
