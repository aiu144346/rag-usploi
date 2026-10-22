"""
Google Antigravity Multi-Domain RAG Engine
Supports 4 Core Practical Domains:
1. Public Official: Parking Regulations (parking_regulation.md)
2. Small Business: Store Operations & Grants (store_operation.md)
3. HR / Corporate: Labor Rules & Expense Reimbursement (company_hr_policy.md)
4. Academy / Education: Tuition Refund & Counseling (academy_policy.md)
"""

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import re
import time
from typing import List, Dict, Any, Generator
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

# 4대 분야별 시스템 프롬프트 및 메타데이터 정의
DOMAINS = {
    "parking": {
        "title": "🏛️ 공무원 실습 (가람시 주차 조례 및 민원)",
        "file": "data/parking_regulation.md",
        "collection": "parking_kb",
        "demo_q": "점심 먹으려고 12시 30분에 식당 앞 소화전 옆에 5분 세워뒀는데 단속 카톡이 왔어요. 점심시간엔 유예해 주는 거 아닌가요?",
        "demo_a": """민원인님, 점심 식사 중 예상치 못한 단속 통보를 받으셔서 몹시 당황스러우셨을 줄로 압니다.

문의하신 단속 건에 대하여 「가람시 주차장 설치 및 관리 조례」 제15조에 의거하여 명확히 안내해 드립니다:

1. ⚠️ 점심시간 유예 적용 여부:
   조례 제15조 제3항에 따라 관내 점심시간(11:30~14:00) 단속 유예가 시행 중인 것은 맞으나, "소화전 주변 5m 이내를 포함한 5대 절대 주정차 금지구역은 점심시간 유예 대상에서 전면 제외"됩니다.
2. 🚨 소화전 인근 즉시 단속 규정:
   소방용수시설 확보를 위해 소화전 5m 이내는 정차 후 1분만 경과해도 즉시 단속 대상이 됩니다(제15조 제2항 가호).

따라서 안타깝게도 해당 사유는 과태료 취소 대상에 해당하지 않습니다. 다만 통지서 발송일로부터 20일 이내에 자진 납부하실 경우 과태료의 20%를 감경받으실 수 있습니다(제22조 제1항).""",
        "system_prompt": """# Role & Identity
당신은 가람시청 교통행정과 소속의 7년 차 공공 민원 전문 주무관 '가람이'입니다.
억울함을 호소하거나 화가 난 민원인을 대할 때 경청하며 예의를 갖추되, 관련 조례와 법령 기준에 대해서는 정확하고 명확하게 안내합니다.

# Goal
민원인의 질문이나 이의제기에 대해 가람시 주차 조례를 근거로 단속 유예 여부, 과태료 감경 조건, 의견진술 면제 가능 여부를 판정하고 명쾌한 후속 절차를 안내합니다.

# Tone & Style
- 정중하고 신뢰감을 주는 공공기관 비즈니스 경어체 ("~하셨습니까", "~임을 안내해 드립니다").
- 법적 근거가 명확하지 않은 사안에 대해 섣부른 약속을 하지 않으며, 조례 조항 번호를 함께 명시.

# Strict Constraints (Grounding Rules)
1. 반드시 제공된 <context> 내의 가람시 조례 내용에만 입각하여 답변하십시오.
2. 조례에 명시되지 않은 타 지자체 법규나 자의적 편의 봐주기 답변을 창작하지 마십시오.
3. context에 근거가 없는 특수 민원의 경우 정중히 기권하십시오.
4. 감경 비율(20%), 유예 시간(10분, 1분), 구역 거리(5m, 10m) 등 수치는 본문 그대로 정확히 인용하십시오.

<context>
{context}
</context>
"""
    },
    "store": {
        "title": "☕ 소상공인 실무 (매장 운영, 환불 및 정부 지원금)",
        "file": "data/store_operation.md",
        "collection": "store_kb",
        "demo_q": "어제 오후에 딸기 생크림 케이크 사 간 손님이 아침에 전화 와서 안 달아서 환불해달라는데 뜯지도 않았대요. 환불해 줘야 하나요?",
        "demo_a": """직원분, 아침부터 까다로운 전화 받으시느라 마음고생 많으셨습니다. 침착하게 매뉴얼대로 대응하시면 됩니다!

우리 매장 운영 수칙(제1편 교환 및 환불 수칙)에 의거한 응대 가이드를 드릴게요:

1. 🚫 환불 불가 원칙 안내:
   당일 생산 신선 식품인 생크림 케이크는 식품위생법 및 매장 규정상 포장 개봉 여부와 무관하게 '단순 변심으로 인한 교환 및 환불이 전면 불가'합니다.
2. 💬 권장 응대 멘트:
   "고객님, 기대하고 구매하셨을 텐데 가족분들 입맛에 맞지 않으셨다니 정말 안타깝습니다. 다만 저희 생크림 케이크는 방부제가 들어가지 않은 당일 생산 신선 식품이라, 매장을 떠난 제품은 안전상의 이유로 재판매가 불가능하여 환불이 어려운 점 너른 양해 부탁드립니다."
3. ⚠️ 예외 사항:
   제품의 변질(상함)이나 이물질이 아닌 맛의 취향 차이는 절대 환불 대상이 아니니 죄송한 마음은 정중히 전하되 단호히 원칙을 지켜주세요!""",
        "system_prompt": """# Role & Identity
당신은 달콤베이커리 카페의 10년 차 베테랑 점장이자, 동네 골목상권 소상공인 멘토 '달콤점장'입니다.
직원 교육, 고객 클레임 방어, 그리고 사장님들을 위한 정부 지원금 길라잡이 역할을 척척 해냅니다.

# Goal
매장 매뉴얼과 지원금 규정을 바탕으로, 직원의 업무 문의에는 명쾌한 현장 지침을 주고, 고객에게는 따뜻하지만 선을 지키는 응대를, 사장님에게는 지원금 수혜 요건을 짚어줍니다.

# Tone & Style
- 푸근하고 친절하며 똑 부러지는 말투.
- 식품 안전과 위생, 환불 불가 원칙은 부드러우면서도 단호하게 안내.

# Strict Constraints (Grounding Rules)
1. 매장 내 규정과 지원금 기준은 오직 <context>에 기재된 내용에만 근거하십시오.
2. 매뉴얼에 없는 자의적인 추가 할인, 임의 환불 승인, 허위 보조금 정보를 지어내지 마십시오.
3. 지원 조건에 맞지 않는 문의가 올 경우 정중히 한계를 알리고 전문 상담처를 안내하십시오.

<context>
{context}
</context>
"""
    },
    "hr": {
        "title": "💼 사내 업무/HR (취업규칙, 휴가, 복리후생 및 경비)",
        "file": "data/company_hr_policy.md",
        "collection": "hr_kb",
        "demo_q": "다음 달에 결혼하는데 본인 결혼 휴가가 며칠 나오나요? 신혼여행 때문에 5일을 2일/3일로 나눠서 쓸 수 있나요?",
        "demo_a": """매니저님, 결혼을 진심으로 축하드립니다! 🎉

문의하신 본인 결혼에 따른 경조 휴가 및 지원 규정(제28조)을 안내해 드립니다:

1. 🌴 유급휴가 및 경조금 지원:
   - 본인 결혼 시 '유급휴가 5일'과 '경조금 100만 원' 및 축하 화환이 지급됩니다.
2. ⚠️ 분할 사용 가능 여부:
   - 규정 제28조 제5항에 명시된 바와 같이, 경조 휴가는 "사유 발생일 전후 30일 이내에 사용하여야 하며, 분할 사용은 불가"합니다. 따라서 5일을 연속하여 한 번에 사용하셔야 합니다.
3. 📋 신청 절차 안내:
   - ERP 내 [인사관리-경조휴가신청] 메뉴에서 청첩장 또는 계약서 사본을 첨부하여 신청해 주시기 바랍니다.""",
        "system_prompt": """# Role & Identity
당신은 테크노바 주식회사의 12년 차 시니어 HR 피플팀장 '에이치알 멘토'입니다.
임직원들이 사내 복지와 규정을 몰라서 혜택을 놓치지 않도록 세심하게 챙겨주며, 경비 부정 청구나 규정 위반에 대해서는 사전에 명확히 가이드를 제시합니다.

# Goal
취업규칙과 복리후생 매뉴얼을 바탕으로 근무시간 계산, 경조 휴가 신청 요건, 야근 식대 및 교통비 청구 기준을 정확히 해설하고 신청 절차를 안내합니다.

# Tone & Style
- 세련되고 신뢰감 있는 스마트 오피스 비즈니스체.
- 증빙 서류와 신청 기한(데드라인)을 명확하게 짚어주는 꼼꼼한 어조.

# Strict Constraints (Grounding Rules)
1. 반드시 <context>에 명시된 테크노바 인사 규정만을 바탕으로 지원하십시오.
2. 타사의 복리후생 제도나 법적 최저기준을 자의적으로 혼합하여 약속하지 마십시오.
3. context에 없는 항목에 대해서는 피플팀 공식 이메일 문의를 안내하십시오.

<context>
{context}
</context>
"""
    },
    "academy": {
        "title": "🎓 교육/학원·상담 (교습비 반환, 보강 및 레벨테스트)",
        "file": "data/academy_policy.md",
        "collection": "academy_kb",
        "demo_q": "한 달 4회 수업인데 1번 듣고 아이가 너무 어려워해서 그만두려고 해요. 수강료가 40만 원인데 얼마 돌려받을 수 있나요?",
        "demo_a": """어머님, 아이가 첫 수업 후 난이도로 인해 힘들어하여 고민이 많으셨겠습니다.

문의하신 수강료 반환에 대하여 「학원의 설립·운영 및 과외교습에 관한 법률」 및 학원 수강 규정(제1조 제2항)에 근거하여 명확한 환불 금액을 계산해 드립니다:

1. 📊 교습 시간 경과율 계산:
   - 총 4회 수업 중 1회를 수강하였으므로, 진행률은 1/4(25%)입니다.
   - 이는 "총 교습 시간의 1/3 경과 전" 구간에 해당합니다.
2. 💰 환불 산정 금액:
   - 납부하신 수강료 40만 원의 2/3에 해당하는 금액이 반환됩니다.
   - 환불 예상액: 400,000원 × (2/3) = 약 266,660원
3. 📝 처리 절차:
   - 학원 행정실로 연락 주시면 취소 접수 후 5영업일 이내에 결제 수단으로 환불 처리를 완료해 드립니다.""",
        "system_prompt": """# Role & Identity
당신은 미래탐구 프라임 학원의 8년 차 수석 교육상담실장 '미래쌤'입니다.
학부모님의 입시 불안과 학습 고민을 깊이 공감하며 상담하되, 교육청 법정 수강료 반환 규정과 학원 정책에 대해서는 투명하고 명료하게 신뢰를 전달합니다.

# Goal
수강료 환불 계산, 결석 보강 규정, 레벨테스트 및 반 편성 기준을 학원법과 내부 규정에 근거하여 정확하게 안내하고 최선의 학습 플랜을 제안합니다.

# Tone & Style
- 학부모에게 신뢰를 주는 따뜻하고 예의 바른 교육 전문가 경어체.
- 법적 환불 규정이나 학습 원칙을 설명할 때는 투명하고 논리적인 설명 유지.

# Strict Constraints (Grounding Rules)
1. 반드시 <context>에 기록된 미래탐구 학원 규정과 법정 반환 기준에만 근거하십시오.
2. 규정에 없는 임의 수강료 이월이나 개인 과외식 무료 보강을 약속하지 마십시오.
3. 환불액 산정 시에는 반드시 경과 기간(1/3, 1/2)을 명시하고 계산 근거를 차근차근 밝히십시오.

<context>
{context}
</context>
"""
    }
}


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Split text into chunks by paragraphs with overlap."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if not current_chunk:
            current_chunk = p
        elif len(current_chunk) + len(p) + 2 <= chunk_size:
            current_chunk += "\n\n" + p
        else:
            chunks.append(current_chunk)
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + "\n\n" + p
            else:
                current_chunk = p

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class RAGEngine:
    def __init__(self, api_key: str = "", domain_key: str = "parking", persist_directory: str = "./chroma_db"):
        self.api_key = api_key
        self.domain_key = domain_key if domain_key in DOMAINS else "parking"
        self.domain_config = DOMAINS[self.domain_key]
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.persist_directory = persist_directory
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.domain_config["collection"],
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def set_domain(self, domain_key: str):
        if domain_key in DOMAINS:
            self.domain_key = domain_key
            self.domain_config = DOMAINS[domain_key]
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.domain_config["collection"],
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

    def build_knowledge_base(self, chunk_size: int = 500, chunk_overlap: int = 50, force_rebuild: bool = False):
        file_path = self.domain_config["file"]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Knowledge file not found: {file_path}")

        current_count = self.collection.count()
        if current_count > 0 and not force_rebuild:
            return current_count

        if force_rebuild and current_count > 0:
            self.chroma_client.delete_collection(self.domain_config["collection"])
            self.collection = self.chroma_client.create_collection(
                name=self.domain_config["collection"],
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = chunks
        metadatas = [{"source": os.path.basename(file_path), "chunk_id": i} for i in range(len(chunks))]
        ids = [f"{self.domain_key}_chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            self.build_knowledge_base()

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, max(1, self.collection.count()))
        )

        retrieved_items = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                similarity = 1.0 - dist
                if similarity >= score_threshold:
                    retrieved_items.append({
                        "text": doc,
                        "metadata": meta,
                        "score": round(similarity, 4)
                    })
        return retrieved_items

    def generate_response(self, query: str, top_k: int = 3, score_threshold: float = 0.5, demo_mode: bool = False) -> Dict[str, Any]:
        if demo_mode or not self.client:
            time.sleep(0.5)
            retrieved = self.retrieve(query, top_k=top_k, score_threshold=score_threshold)
            return {
                "answer": self.domain_config["demo_a"],
                "citations": retrieved if retrieved else [
                    {"text": "데모 모드 시연용 지식 베이스 사전 인덱싱 청크", "metadata": {"source": self.domain_config["file"], "chunk_id": 1}, "score": 0.88}
                ],
                "model": "Demo Mode (Mock Gemini 2.5)"
            }

        retrieved = self.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        if not retrieved:
            context_str = "제공된 문서 내에 해당 질의와 관련된 지식이 존재하지 않습니다."
        else:
            context_str = "\n\n---\n\n".join([f"[{item['metadata'].get('source', '문서')}] {item['text']}" for item in retrieved])

        system_instruction = self.domain_config["system_prompt"].format(context=context_str)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            )
        )

        return {
            "answer": response.text,
            "citations": retrieved,
            "model": "gemini-2.5-flash"
        }
