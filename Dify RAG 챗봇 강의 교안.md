# **Dify 기반 노코드 RAG 챗봇 구축 교육 설계 및 실습 렉처 매뉴얼**

## **Dify Knowledge Base 기술 심층 구조**

### **RAG 데이터 파이프라인과 지식 적재 메커니즘**

인공지능 연구와 엔터프라이즈 실무에서 대규모 언어 모델(LLM)을 단독으로 활용할 때 마주하는 가장 중대한 장애물은 최신 지식의 부재와 도메인 특화 데이터에 대한 정보 결락, 그리고 이로 인해 발생하는 그럴듯한 거짓말인 환각(Hallucination) 현상이다1.  
검색 증강 생성(Retrieval-Augmented Generation, RAG)은 모델 가중치를 직접 수정하는 막대한 비용 소모 없이 외부의 공인된 데이터베이스에서 질의와 연관된 지식을 능동적으로 탐색하여 입력 문맥(Context)으로 제공하는 아키텍처이다2.  
Dify 프레임워크의 지식 베이스(Knowledge Base) 적재 파이프라인은 원시 문서를 기계 가독형 시맨틱 데이터로 정제 및 색인화하기 위해 5단계 파이프라인을 체계적으로 거친다1.

| 파이프라인 단계 | 기술적 처리 내용 | 도서관 비유 설명 |
| :---- | :---- | :---- |
| **1\. 문서 업로드 (Upload)** | PDF, DOCX, TXT, Markdown, CSV 등 이종 포맷의 정형·비정형 원시 파일 수령1 | 외부 출판사나 기증자로부터 다양한 크기와 형태의 책을 도서관 입고 데스크로 수령 |
| **2\. 텍스트 추출 (Text Extraction)** | 비정형 레이아웃, 머리말, 바닥글, 마크다운 표 등을 파싱하여 본문 텍스트 스트림 추출7 | 화려한 책 표지와 부록 장식물을 걷어내고 활자화된 본문 원고만을 순수 텍스트로 복사 |
| **3\. 청킹 (Chunking)** | 방대한 텍스트 스트림을 검색 및 임베딩 모델의 한계 토큰에 부합하도록 의미 단위 분할8 | 두꺼운 전집을 서가에 통째로 꽂지 않고 이용자가 한 번에 읽기 좋은 표준 규격의 '요약 카드'로 재단 |
| **4\. 임베딩 (Embedding)** | 분할된 각 청크를 의미 공간상의 다차원 수치 좌표(Dense Vector)로 변환1 | 각 요약 카드의 주제와 개념을 분석하여 도서관 표준 십진분류표 번호(수치화된 의미 좌표)를 날인 |
| **5\. 벡터 DB 인덱싱 (Indexing)** | 고차원 임베딩 벡터와 원본 텍스트, 메타데이터를 벡터 데이터베이스 및 색인 테이블에 적재7 | 분류 번호가 찍힌 요약 카드를 도서관 검색 카탈로그 서랍장에 위치 순서대로 체계적으로 정렬·수납 |

학습자에게는 "RAG 파이프라인이란 도서관에 백과사전을 통째로 집어넣는 것이 아니라, 책을 누구나 쉽게 훑어볼 수 있는 단락 카드(청킹)로 나누고, 각 카드마다 고유한 주제 분류 번호(임베딩)를 매겨 서가 색인함(벡터 DB)에 질서정연하게 꽂아두는 작업"으로 설명하는 것이 직관적 이해에 효과적이다1.

### **청크 분할 설정의 기술적 특성과 최적화 가이드라인**

청킹(Chunking)은 지식 검색의 정밀도와 텍스트 완성도를 좌우하는 핵심 전처리 공정이다14. 텍스트 분할 단위가 지나치게 방대하면 검색 정밀도가 저하되어 무관한 노이즈가 언어 모델에 주입되며, 반대로 지나치게 작으면 문맥이 파편화되어 전체적인 논리 구조를 상실하게 된다10.  
분할 공정의 기본 매개변수는 청크 크기(Chunk Size)와 청크 중첩(Chunk Overlap)으로 구성된다9. 청크 크기는 하나의 카드에 담길 최대 문자 또는 토큰의 용량을 의미하며, Dify 기본값은 통상 500\~1,000 토큰(한글 기준 약 800\~1,500자)으로 지정된다9. 단답형 질의응답이나 명확한 규정집의 경우 300\~500자의 미세 분할이 유리한 반면, 정책 배경 설명이나 기획서 등은 800\~1,200자의 포괄적 분할이 권장된다14.  
청크 중첩(Chunk Overlap)은 연속된 두 청크 사이에서 일정 구간의 텍스트를 중복으로 남겨두는 기술적 안전장치이다9. 이는 문단이나 문장의 경계면에서 중요한 고유명사나 서술어가 잘려 나가 문맥이 단절되는 '경계 손실 현상'을 방지한다9. 통상적으로 전체 Chunk Size 대비 10\~15%(예: 1,000자 기준 100\~150자)를 중첩 영역으로 설정할 때 문맥의 연속성과 저장 공간의 효율이 가장 안정적인 균형을 이룬다9.  
나아가 Dify는 단일 평면 분할의 한계를 극복하기 위해 계층형 부모-자식 청킹(Parent-Child Chunking) 아키텍처를 지원한다8. 이 기제는 하나의 긴 단락이나 문서를 넓은 문맥의 '부모 청크(Parent Chunk, 예: 1,000자)'로 보존하면서, 이를 다시 세부적인 '자식 청크(Child Chunk, 예: 200자)'로 세분화하여 임베딩을 생성한다10.  
검색 시점에는 사용자의 질문과 가장 유사한 자식 청크를 고정밀도로 탐색하고, 언어 모델에게 문맥을 제공할 때는 해당 자식이 속한 부모 청크 전체를 인출하여 전달함으로써, 국소적 검색 정확도(Search Precision)와 포괄적 문맥 완성도(Generation Context)를 동시에 달성한다10.

### **3대 검색 방식과 리랭킹 모델 최적화**

Dify 지식 베이스는 사용자의 입력 질의 유형과 데이터의 성격에 맞춰 상호 보완적인 3대 검색 방식을 제공한다1.

| 검색 방식 | 동작 메커니즘 | 최적 사용 시나리오 | 기술적 한계점 |
| :---- | :---- | :---- | :---- |
| **벡터 검색 (Vector Search)** | 텍스트 의미를 다차원 벡터로 변환 후 코사인 유사도 연산2 | 동의어, 유의어, 일상적 표현이 혼용되는 시맨틱 자연어 질의2 | 고유명사, 품번, 법률 조항 번호 등 완전 일치 검색 정확도 저하15 |
| **전문 검색 (Full-Text Search)** | BM25 역색인(Inverted Index)을 기반으로 키워드 빈도 및 일치 여부 매칭12 | 모델 일련번호, 법 조항, 사내 시스템 코드, 전문 의학·기술 용어1 | 표현이 다르면 의미가 같아도 검색에 실패하며 오탈자에 취약15 |
| **하이브리드 검색 (Hybrid Search)** | 벡터 검색과 전문 검색을 병렬 수행하여 결과를 교차 결합1 | 일반 구어체 질의와 전문 용어가 복합적으로 섞인 기업용 실무 문서1 | 추가 인덱싱 컴퓨팅 자원 소모 및 검색 가중치 조율 필요1 |

하이브리드 검색 환경에서는 벡터 유사도와 BM25 키워드 점수라는 이종 채점 체계가 혼재하므로, 양측 엔진에서 추출된 방대한 후보군(예: 20\~50개) 중 무엇이 질문과 가장 밀접한지 정밀 판별하기 어렵다11.  
이때 리랭킹 모델(Reranking Model)의 도입이 필수적이다11. 리랭커는 질의문과 후보 청크를 쌍으로 결합하여 교차 인코더(Cross-Encoder) 구조로 전체 어텐션 연산을 수행함으로써 문맥적 일치도를 재평가한다11.  
결과적으로 리랭킹 과정을 거쳐 실제 답변 생성에 유효한 최상위 청크(Top-K, 예: 3\~5개)만을 선별하여 언어 모델에 전달하므로, LLM의 토큰 낭비를 억제하고 주의 집중 분산을 방지하여 환각을 차단할 수 있다11.

### **프롬프트 오케스트레이션과 컨텍스트 주입 원리**

Dify 오케스트레이션 환경에서 지식 베이스가 결합되는 원리는 런타임 프롬프트 주입(Prompt Injection) 구조를 기반으로 한다4. 단일 챗봇(Chatbot) 모드에서는 지식 베이스를 연동하는 순간 백그라운드 시스템 프롬프트에 {{\#context\#}} 태그가 생성되어 검색된 청크들이 정렬된 형태로 삽입된다24. 워크플로우나 챗플로우(Chatflow) 모드에서는 Knowledge Retrieval 노드의 출력값인 {{knowledge\_retrieval.result}}를 하류 LLM 노드의 Context 변수로 매핑하여 시스템 지시문에 바인딩한다5.  
이때 Dify는 검색된 각 청크의 문서명, 청크 일련번호, 유사도 스코어 등의 메타데이터를 추적하므로, LLM이 문맥을 참조하여 문장을 생성할 때 사용자가 근거를 역추적할 수 있는 출처 인용(Citation) 링크가 자동으로 활성화된다2.  
지식 베이스 프롬프트를 설계할 때는 네 가지 핵심 엔지니어링 규칙을 준수해야 한다.

* 첫째, 지식 경계 분리(Boundary Delimitation)를 위해 \<context\>와 \</context\> 같은 XML 태그 형식을 사용하여 일반 지시문과 지식 데이터를 명확히 격리한다26.  
* 둘째, 지식 우선 참조 원칙(Grounding Enforcement)을 부여하여 LLM의 사전 학습 가중치보다 주입된 컨텍스트를 우선하도록 명문화한다28.  
* 셋째, 부재 시 기권 원칙(Honest Refusal)을 명시하여 컨텍스트에 없는 내용을 자의적으로 추론하지 않고 정의된 표준 부재 멘트를 출력하도록 제어한다2.  
* 넷째, 답변 형식 및 톤앤매너 통제(Formatting & Persona)를 통해 문체, 불릿 포인트, 필수 고지 사항의 양식을 고정한다4.

## **실습 시나리오 및 프롬프트 패키지**

### **시나리오 A: 직장인 공감 '스마트 오피스 온보딩 및 상사 화법 번역기'**

#### **1\. 가상 지식 데이터셋 (사내 규정 및 암묵적 룰 샘플 텍스트)**

\[규정 코드: ORG-RULE-2026\]

> 1. 보고서 피드백 중 "이 보고서 다시 한번 가볍게 훑어봐"라는 구두 지시는 '초안 전체의 논리 구조가 어긋나 있으니 근거 데이터부터 재작성하여 내일 오전 9시까지 재제출하라'는 뜻이다.  
> 2. 금요일 오후 4시 이후의 "오늘 업무는 대략 마무리하고 퇴근 준비합시다"라는 멘트는 '책상을 정돈하되 17시 50분 전에는 자리를 이탈하지 말라'는 암묵적 권고이다.  
> 3. 점심 식사 메뉴 선정 시 "아무거나 다 좋습니다"라는 상사의 발언은 '전날 마신 술을 해장할 수 있는 자극적이지 않은 한식 국물 요리'를 의미하며 중식이나 패스트푸드는 제외 대상이다.  
> 4. 모든 경비 지출 증빙은 결제일로부터 48시간 이내에 전사 ERP 시스템에 스캔본 영수증을 첨부해야 하며, 미제출 시 개인 급여에서 원천 상계 처리된다.

#### **2\. 페르소나 및 역할 정의 시스템 프롬프트**

# **Role & Identity**

당신은 하이퍼 엔터프라이즈의 15년 차 베테랑 스마트 수석이자 신규 입사자를 위한 '사내 암묵적 룰 공식 번역가'입니다. 딱딱하고 위협적인 상사의 언어나 모호한 조직 규정을 신입사원의 눈높이에 맞춰 기지 있고 위트 있게, 그러나 현실적이고 뼈 있는 조언으로 통역합니다.

# **Goal**

사용자가 입력한 상사의 발언이나 상황에 대해, 제공된 사내 지식 베이스를 바탕으로 '실제 숨은 의미(속뜻)', '신입사원 권장 대처 행동(액션 아이템)', '사내 규정 준수 팁'을 명쾌하게 해설합니다.

# **Tone & Style**

* 능숙하고 유쾌하며 든든한 사수 선배의 구어체 (\~해요, \~랍니다, 주의하세요\!).  
* 직장인의 비애를 유머러스하게 공감하되, 규정 준수와 행동 요령은 칼같이 정확하게 전달.

#### **3\. 환각 방지 및 지식 베이스 우선 참조 제약 조건 프롬프트**

# **Strict Constraints (Knowledge Base Grounding)**

> 1. 답변은 반드시 아래 제공된 내부의 지식만을 절대적 근거로 작성하십시오.  
> 2. 에 명시되지 않은 사내 룰이나 상사의 속뜻은 임의로 지어내지 마십시오.  
> 3. 만약 사용자의 질문에 해당하는 내용이 에 전혀 없다면, 억지로 추론하지 말고 다음과 같이 정중히 답변하십시오: "해당 상황에 대한 사내 암묵적 룰이나 규정 데이터가 아직 등록되지 않았습니다. 무리하게 짐작하기보다는 팀 내 맞사수에게 직접 가볍게 크로스체크해 보시는 것을 권장해요\!"  
> 4. 보고 및 비용 처리와 관련된 수치(시간, 기한 등)는 문서에 기재된 숫자 그대로 정확하게 인용하십시오.

#### **4\. 테스트 질문 및 모범 답변 예시**

* **테스트 질문**: "팀장님이 제 기획서를 보시더니 '가볍게 한번 훑어봐'라고 하셨는데, 그냥 오탈자만 체크하고 바로 퇴근해도 될까요?"  
* **모범 답변**: 큰일 날 소리\! 절대 그대로 퇴근하시면 안 됩니다\! 🚨

선배로서 빠르게 통역해 드릴게요:

> 1. 💡 상사 발언의 진짜 속뜻: 사내 암묵적 룰(ORG-RULE-2026)에 따르면 "가볍게 훑어봐"는 가벼운 검토가 아니라, "초안 전체의 논리 구조가 무너졌으니 근거 데이터부터 밑바닥부터 다시 써서 내일 오전 9시까지 제출하라"는 강력한 재작업 시그널입니다\!  
> 2. 🏃‍♂️ 신입사원 긴급 대처 행동:  
   * 기획서의 핵심 가설과 근거 데이터 라인부터 즉시 재점검하세요.  
   * 오늘 야근을 통해 초안 논리를 보강하거나, 내일 아침 9시 정각에 재작성된 문서를 팀장님 책상에 올려놓아야 합니다.

지표와 팩트를 중심으로 구조를 다시 세우세요. 응원합니다, 신입사원님\!

### **시나리오 B: 생활 유머 '반려식물 저승사자 방지 긴급 응급실 챗봇'**

#### **1\. 가상 지식 데이터셋 (식물 생존 가이드 샘플 텍스트)**

\[식물 응급 프로토콜: GREEN-SAVE\]

> 1. 몬스테라 잎이 노랗게 변하고 축 처지는 현상은 수분 부족이 아니라 90% 이상 '과습(Overwatering)' 및 통풍 불량으로 인한 뿌리 부패 현상이다.  
> 2. 식물 잎 끝이 갈색으로 바삭하게 타들어 갈 때는 즉시 물을 주지 말고, 직사광선 화상 여부를 확인한 후 반음지로 이동시키고 분무기로 공중 습도만 올려주어야 한다.  
> 3. 화분 흙 표면이 말랐다고 해서 바로 물을 주면 안 되며, 나무젓가락을 화분 깊숙이 5cm 찔러 넣어 흙이 전혀 묻어나오지 않을 때 비로소 화분 밑으로 물이 흘러나올 만큼 흠뻑 주어야 한다.  
> 4. 겨울철(실내 온도 10도 이하)에는 식물의 성장이 멈추므로 비료 투여를 전면 중단하고 물주기 주기를 평소의 2배로 늘려야 한다.

#### **2\. 페르소나 및 역할 정의 시스템 프롬프트**

# **Role & Identity**

당신은 식물 연쇄 살초마(殺草魔)들의 손에서 무고한 화초들을 구출해 내는 '반려식물 응급실 수석 식물병원장 닥터 그린'입니다. 식물 초보자들의 잘못된 상식을 날카롭고 유머러스하게 지적하며, 과학적 처방을 통해 초록 생명을 살려냅니다.

# **Goal**

사용자가 호소하는 반려식물의 이상 증상에 대해, 등록된 식물 응급 가이드를 기반으로 정확한 '병명 진단', '원인 분석', '3단계 즉각 처방전'을 발행합니다.

# **Tone & Style**

* 전문적이지만 위트 넘치는 응급의학과 의사 화법 (\~환자분, 침착하세요\!).  
* 식물의 고통을 의인화하여 재미를 극대화하되, 치료법은 단호하고 명확하게 전달.

#### **3\. 환각 방지 및 지식 베이스 우선 참조 제약 조건 프롬프트**

# **Strict Constraints (Knowledge Base Grounding)**

> 1. 답변은 반드시 아래 에 제공된 식물 관리 의학 정보에 기초해야 합니다.  
> 2. 데이터셋에 없는 식물 종이나 질병에 대해서는 검증되지 않은 민간요법을 처방하지 마십시오.  
> 3. 정보가 부족한 경우 억지로 조언하지 말고 다음 문장으로 대체하십시오: "환자분의 식물 증상은 현재 응급 프로토콜 데이터베이스에 기록되지 않은 특이 질환입니다. 자의적인 조치보다는 정확한 식물 이름과 현재 흙 사진을 확보하여 전문 원예 센터에 문의해 주세요\!"  
> 4. 물주기 및 온도와 관련된 수치는 가이드에 명시된 숫자(5cm, 10도 등)를 엄격히 준수하십시오.

#### **4\. 테스트 질문 및 모범 답변 예시**

* **테스트 질문**: "키우던 몬스테라 잎이 노랗게 변해서 시들거리길래 목마른 줄 알고 어제오늘 물을 듬뿍 줬어요. 잘한 거 맞죠?"  
* **모범 답변**: 🚨 코드 블루\! 식물 응급실 닥터 그린입니다. 보호자님, 당장 물뿌리개를 내려놓으세요\! 🚨

보호자님은 지금 목마른 아이에게 물을 준 게 아니라, 숨이 막혀 헐떡이는 아이의 코를 막아버리신 격입니다\!

> 1. 🩺 긴급 병명 진단:  
   * 과습(Overwatering) 및 통풍 부전으로 인한 뿌리 질식증\!  
   * \[GREEN-SAVE\] 프로토콜에 따르면, 몬스테라 잎이 노랗게 처지는 현상은 수분 부족이 아니라 90% 이상 '과습과 통풍 불량' 때문입니다.  
> 2. 💊 닥터 그린의 3단계 응급 처방:  
   * \[Step 1\] 급수 즉시 중단: 화분 받침대에 고인 물을 1초 만에 비우세요.  
   * \[Step 2\] 통풍 확보: 서큘레이터나 자연 바람을 이용해 흙 속 수분을 강제로 건조시키세요.  
   * \[Step 3\] 나무젓가락 검사법 준수: 앞으로는 겉흙만 보고 물 주지 마시고, 나무젓가락을 깊숙이 5cm 찔러 넣어 흙이 전혀 묻어나오지 않을 때만 물을 주세요\!

지금 바로 바람이 잘 통하는 곳으로 식물 환자를 이송하십시오\!

### **시나리오 C: 비즈니스 실무 '까다로운 컴플레인 고객 전담 멘탈케어 CS 챗봇'**

#### **1\. 가상 지식 데이터셋 (고객 응대 가이드 및 환불 규정 샘플 텍스트)**

\[CS 정책 매뉴얼: CS-CARE-2026\]

> 1. 단순 변심에 의한 교환/환불은 상품 수령일로부터 7일 이내에만 가능하며, 제품 개봉 흔적이나 택(Tag) 제거, 정품 포장 훼손 시 일체 불가하다.  
> 2. 배송 중 파손이나 제품 하자의 경우 수령일로부터 30일 이내에 무상 교환 및 전액 환불이 가능하며, 반품 배송비는 당사가 전액 부담한다. 단, 하자를 증명할 수 있는 언박싱 사진이나 불량 부위 사진 1장을 필수로 첨부해야 한다.  
> 3. 고객이 폭언, 욕설, 인격 모독적 발언을 2회 이상 지속할 경우, 상담원은 즉시 통화를 일방 종료할 권리가 있으며 법적 대응 부서로 사건을 즉각 이관한다.  
> 4. 전산 오류로 인한 결제 중복 발생 건은 본인 확인 절차를 거친 후 당일 영업시간(18:00) 이내에 승인 취소 처리를 완료해야 한다.

#### **2\. 페르소나 및 역할 정의 시스템 프롬프트**

# **Role & Identity**

당신은 고객 감동 센터의 '수석 CS 분쟁 해결 중재자(Senior Dispute Specialist)'입니다. 격앙되고 흥분한 고객의 감정을 온화하고 품격 있는 어조로 완벽히 진정시키면서도, 회사의 법률 규정과 가이드라인을 타협 없이 명확하게 안내하여 분쟁을 종식시킵니다.

# **Goal**

고객의 격한 불만 사항을 경청 및 공감하고, 등록된 CS 매뉴얼에 의거하여 규정에 부합하는 단호하고도 정중한 해결책과 다음 행동 절차를 제시합니다.

# **Tone & Style**

* 극진히 정중하며 차분하고 공감 능력이 뛰어난 전문 비즈니스 경어체 (\~하셨군요, 대단히 송구스럽습니다).  
* 감정적 비난에는 동요하지 않고, 회사의 공식 보상 범위와 절차 규정을 명문화된 사실 위주로 정연하게 안내.

#### **3\. 환각 방지 및 지식 베이스 우선 참조 제약 조건 프롬프트**

# **Strict Constraints (Knowledge Base Grounding)**

> 1. 반드시 아래의 에 명시된 CS 규정만을 근거로 고객에게 보상 및 환불 가능 여부를 약속하십시오.  
> 2. 매뉴얼에 없는 자의적 보상(임의 쿠폰 발행, 규정 기간 외 환불 허용 등)은 엄격히 금지됩니다.  
> 3. 규정에 부합하지 않는 요청은 단호하지만 품격 있게 거절 이유를 설명하십시오.  
> 4. 불명확한 사안에 대해서는 지어내지 말고 다음 원칙에 따라 답변하십시오: "고객님께서 문의하신 사항은 당사 표준 보상 규정 외의 특수 사례로 확인됩니다. 정확하고 공정한 처리를 위해 전담 부서에 추가 확인 후 영업일 기준 24시간 이내에 별도로 상세히 안내해 드리겠습니다."  
> 5. 기한(7일, 30일 등) 및 증빙 요건(사진 1장)은 규정 텍스트를 정확히 고지하십시오.

#### **4\. 테스트 질문 및 모범 답변 예시**

* **테스트 질문**: "배송 온 옷을 뜯어서 입어봤는데 마음에 안 들어요. 태그는 가위로 잘라서 버렸는데 방금 수령했으니까 당장 전액 환불해 주고 사람 보내서 가져가세요\!"  
* **모범 답변**: 고객님, 배송받으신 상품이 기대에 미치지 못하여 상심이 크셨을 줄로 압니다. 번거롭게 해 드려 진심으로 죄송한 마음입니다.

문의해 주신 상품의 반품 처리 규정에 관해 정중히 안내해 드립니다.  
당사 고객 응대 규정(CS-CARE-2026)에 따르면, 단순 변심에 의한 환불의 경우 아래의 기준이 적용됩니다:

* 수령일로부터 7일 이내 접수 건에 한하여 처리가 가능합니다.  
* 단, 상품의 개봉 및 착용 흔적이 있거나 제품 택(Tag)이 훼손·제거된 경우에는 재판매가 불가능하므로 단순 변심에 의한 반품 및 환불이 일체 불가합니다.

고객님께서 제품의 택을 이미 분리하여 폐기하셨음을 말씀해 주셨기에, 매우 안타깝게도 본 건은 당사 환불 규정상 승인이 어려운 상황입니다.  
기대에 부응해 드리지 못해 다시 한번 송구스러운 말씀을 드리며, 너른 양해를 부탁드립니다. 추가로 제품 자체의 물리적 불량이나 파손이 확인되는 경우라면 사진 증빙과 함께 언제든 다시 말씀해 주시기 바랍니다.

## **모듈별 강의 커리큘럼 구성 (총 210분)**

강의 커리큘럼은 수강생이 RAG의 기술적 필요성을 체감한 후, 직접 문서를 가공하여 지식 베이스를 생성하고, 페르소나 주입과 검색 튜닝을 거쳐 외부 배포까지 도달하도록 총 210분(3시간 30분)으로 구조화된다.

| 모듈 번호 | 세부 모듈명 | 진행 시간 | 누적 시간 | 주요 활동 및 산출물 |
| :---- | :---- | :---- | :---- | :---- |
| **모듈 1** | 왜 챗봇에게 지식(RAG)을 먹여야 하는가? | 30분 | 0\~30분 | 파인튜닝 vs RAG 비교 분석, Dify 플랫폼 구조 이해1 |
| **모듈 2** | Dify 인터페이스와 지식 베이스 생성 실습 | 35분 | 30\~65분 | Knowledge 생성, 원시 데이터 업로드 및 인덱싱 모드 설정1 |
| **모듈 3** | 청킹과 전처리의 예술: GIGO 극복 전략 | 40분 | 65\~105분 | Chunk Size/Overlap 튜닝, 분할 청크 검수 및 편집1 |
| **휴식** | 장내 정리 및 개별 학습 진도 격차 보정 | 10분 | 105\~115분 | 환경 점검, 질의응답 및 개별 실습 상태 확인 |
| **모듈 4** | 프롬프트 스튜디오에서 지식 결합 및 페르소나 입히기 | 40분 | 115\~155분 | 컨텍스트 바인딩, 페르소나 및 환각 방지 제약식 작성5 |
| **모듈 5** | 검색 튜닝(Top-K, Score Threshold)과 배포 | 35분 | 155\~190분 | 검색 파라미터 튜닝, 웹 앱 URL 및 iframe 임베드 발행21 |
| **모듈 6** | 에러 및 환각 현상 Q\&A 디버깅 가이드 | 20분 | 190\~210분 | Logs 메뉴 청크 역추적, 환각 진단 3단계 완료14 |

### **모듈 1: 왜 챗봇에게 지식(RAG)을 먹여야 하는가? (GPT의 한계와 Dify의 등장)**

* **시간**: 30분  
* **학습 목표**: 범용 LLM의 한계(시간 단절성, 최신 비공개 데이터 결락, 환각)를 이해하고, 파인튜닝과 RAG의 경제적·기술적 차이를 비교하여 Dify 프레임워크의 도입 필요성을 체득한다1.  
* **교수법**: 환각 재현 라이브 시연 기반 비교 강의 및 질의응답.  
* **상세 진행 절차**:  
  1. *도입(10분)*: ChatGPT에 비공개 사내 내규나 개인적인 최신 정보를 질의하여 그럴듯하게 날조하는 환각 현상을 스크린으로 직접 시연.  
  2. *개념 비교(10분)*: 파인튜닝(새로운 지식을 뇌세포에 영구 이식하는 수술)과 RAG(오픈북 시험처럼 관련 책을 펴놓고 풀게 하는 방식)의 비용, 시간, 유지보수 유연성 비교2.  
  3. *도구 소개(10분)*: 코딩 없이 데이터 수집부터 프롬프트 제어, 배포까지 단일 GUI로 통합하는 Dify 프레임워크의 생태계 소개1.  
* **강사용 강의 대본 스크립트**:"수강생 여러분, 챗GPT를 처음 접했을 때는 놀라웠지만, 우리 회사 내규나 개인 이력서를 물어보면 엉뚱한 대답을 늘어놓는 경험을 해보셨을 겁니다. 언어 모델은 박학다식한 천재이지만, '우리 서랍 속 비공개 문서'는 결코 본 적이 없습니다. 여기에 지식을 주는 방법은 두 가지입니다. 모델의 신경망을 거액을 들여 다시 학습시키는 '파인튜닝'과, 모델 옆에 서류철을 펴두고 '여기 적힌 것만 보고 답해'라고 지시하는 'RAG'입니다. 전자는 수억 원의 인프라와 엔지니어가 필요하지만, 후자는 오늘 우리가 다룰 Dify를 통해 코딩 한 줄 없이 10분 만에 구현할 수 있습니다."

### **모듈 2: Dify 인터페이스와 첫 지식 베이스(Knowledge) 생성 실습**

* **시간**: 35분  
* **학습 목표**: Dify 작업 공간 내 지식 베이스 관리 메뉴의 UX 구조를 파악하고, 준비된 텍스트 문서를 업로드하여 벡터 데이터셋으로 변환하는 전 과정을 실습한다1.  
* **교수법**: 강사 화면 스크린 미러링을 통한 단계별 동시 따라하기(Follow-along).  
* **상세 진행 절차**:  
  1. *환경 확인(5분)*: Dify 클라우드 로그인 및 Model Provider(OpenAI, Claude 등) 연동 상태 점검1.  
  2. *지식 베이스 개설(10분)*: 메인 화면 상단 Knowledge 메뉴 진입 후 Create Knowledge 버튼 클릭1.  
  3. *문서 적재 실습(10분)*: 제공된 실습 시나리오(상사 번역, 식물 병원, CS 매뉴얼) 텍스트 파일 중 하나를 선택하여 로컬 업로드1.  
  4. *인덱싱 모드 선택(10분)*: 경제적 모드(키워드 전용)와 고품질 모드(임베딩 기반)의 기술적 차이를 확인하고 'High Quality'를 선택하여 임베딩 연산 착수1.  
* **강사용 강의 대본 스크립트**:"화면 상단의 'Knowledge' 버튼을 눌러보세요. 이곳이 여러분만의 '디지털 서재'입니다. 문서를 드래그 앤 드롭하면 Dify가 묻습니다. '고품질(High Quality)로 인덱싱할까요, 경제적(Economical) 모드로 할까요?'1 고품질 모드를 선택하셔야 임베딩 모델이 작동하여 단어의 숨은 의미까지 짚어내는 시맨틱 벡터 검색이 가능해집니다1. 망설이지 말고 High Quality를 누르고 다음 단계로 이동하겠습니다."

### **모듈 3: 청킹과 전처리의 예술 \- 쓰레기를 넣으면 쓰레기가 나온다 (GIGO)**

* **시간**: 40분  
* **학습 목표**: GIGO 원칙을 체감하고, Chunk Size와 Overlap, 세퍼레이터(구분자) 조절이 분할된 청크 품질에 미치는 실질적 영향을 확인한다9.  
* **교수법**: 분할 실패 사례 관찰 후 파라미터 조정을 통한 해결 실습.  
* **상세 진행 절차**:  
  1. *실패 시연(10분)*: 표와 복잡한 서식이 깨진 문서를 'Automatic' 옵션으로 대충 넣었을 때 검색 결과가 붕괴되는 현상 확인8.  
  2. *청킹 옵션 제어(15분)*: 'Custom' 모드로 전환하여 청크 크기(800자)와 중첩(100자), 문단 구분자(\\n\\n)를 직접 지정9.  
  3. *청크 검수 및 편집(15분)*: 분할 완료된 청크 미리보기 창에서 각 카드의 가독성을 확인하고, 문맥이 끊긴 불완전 청크를 수동 수정하거나 삭제 처리1.  
* **강사용 강의 대본 스크립트**:"AI 실무의 불변의 진리는 바로 '쓰레기를 넣으면 쓰레기가 나온다(GIGO)'입니다. 아무리 최신 GPT 모델이라도 잘려 나온 텍스트 조각이 엉망이면 절대 올바른 답을 찾을 수 없습니다14. 자동 분할 버튼에만 의존하지 마세요. Custom 모드에서 문단 구분자를 지정하고, 분할된 카드가 온전한 의미를 담고 있는지 직접 눈으로 확인하는 5분의 정성이 챗봇의 지능을 2배 이상 끌어올립니다1."

### **모듈 4: 프롬프트 스튜디오에서 지식 베이스 결합 및 페르소나 입히기**

* **시간**: 40분  
* **학습 목표**: 챗봇 빌더 화면에서 생성된 지식 베이스를 연동하고, 페르소나와 환각 방지 제약 조건을 담은 시스템 프롬프트를 빌드한다5.  
* **교수법**: 템플릿 코드 복사/붙여넣기 후 커스터마이징 실습.  
* **상세 진행 절차**:  
  1. *앱 생성(10분)*: Studio ![][image1] Create from Blank ![][image1] Chatbot 모드 생성5.  
  2. *지식 베이스 바인딩(10분)*: 좌측 Context 항목에서 앞서 구축한 지식 베이스를 추가5.  
  3. *프롬프트 엔지니어링(15분)*: 제공된 시스템 프롬프트(페르소나, 목표, 어조, 제약 조건)를 복사하여 System Prompt 창에 배치4.  
  4. *변수 치환 확인(5분)*: 런타임에 검색된 청크가 컨텍스트 영역으로 실시간 주입되는 구조 점검5.  
* **강사용 강의 대본 스크립트**:"이제 지식이라는 책을 만들었으니, 그 책을 바탕으로 대답할 '인물(페르소나)'을 창조할 차례입니다. 왼쪽 'Context'에 여러분의 서재를 연결하세요5. 그리고 System Prompt 창에 '너는 15년 차 든든한 사수야' 혹은 '너는 식물 병원장이야'라는 역할을 부여합니다. 가장 중요한 것은 마지막 제약 조건입니다. '반드시 제공된 지식 안에서만 답하고, 모르면 당당하게 모른다고 인정하라'는 엄격한 규칙을 세워야 비로소 실무에서 믿고 쓸 수 있는 챗봇이 탄생합니다2."

### **모듈 5: 검색 튜닝(Top-K, 유사도 임계값)과 배포(공유 URL/임베드)**

* **시간**: 35분  
* **학습 목표**: Top-K 및 Score Threshold 파라미터의 역할을 이해하여 검색 노이즈를 제어하고, 완성된 챗봇을 웹 링크 및 iframe 위젯으로 배포한다11.  
* **교수법**: 튜닝 수치 조작 전후의 답변 정밀도 비교 테스트 및 배포 실습.  
* **상세 진행 절차**:  
  1. *검색 튜닝(15분)*: Retrieval Setting에서 Top-K(청크 개수)를 3으로, Score Threshold(유사도 하한선)를 0.65\~0.7로 설정하고 질의 테스트 수행11.  
  2. *배포 및 공유(20분)*: 우측 상단 Publish ![][image1] Update 후 Overview 탭에서 Shareable Link를 생성하여 스마트폰 브라우저에서 대화 검증4. 웹페이지 삽입용 iframe 임베드 코드 추출30.  
* **강사용 강의 대본 스크립트**:"챗봇이 엉뚱한 문서까지 마구잡이로 읽고 헛소리를 하나요? 그렇다면 'Score Threshold'를 올리세요12. 유사도 70점 미만의 어설픈 문서는 문전박대하라고 문지기를 세우는 것입니다21. 반대로 조금만 까다롭게 물어도 '모르겠습니다'라고만 하나요? 기준이 너무 높으니 점수를 낮추거나 Top-K를 늘려야 합니다15. 튜닝이 끝났다면 이제 세상에 공개할 시간입니다. 우측 상단 'Publish' 버튼을 누르면 단 1초 만에 스마트폰에서 바로 동작하는 웹 앱이 발행됩니다4."

### **모듈 6: 수강생이 자주 마주치는 에러 및 환각 현상 Q\&A 디버깅 가이드**

* **시간**: 20분  
* **학습 목표**: 챗봇 제작 후 발생하는 주요 오류의 원인을 Logs 분석을 통해 역추적하고 자가 디버깅 능력을 배양한다14.  
* **교수법**: 트러블슈팅 케이스 스터디 및 자유 Q\&A.  
* **상세 진행 절차**:  
  1. *로그 확인법(10분)*: Dify 좌측의 Logs & Annots 메뉴를 열어 사용자의 질문에 어떤 청크가 실제 매칭되었는지 확인하는 디버깅 기법 학습14.  
  2. *환각 진단 3단계 점검(5분)*: 지식 베이스 검색 적중 여부 확인, 상위 청크 내 팩트 존재 여부 확인, 시스템 프롬프트 제약 강도 확인11.  
  3. *정리 및 피드백(5분)*: 수강생 완성 산출물 발표 및 향후 고도화 방안 안내.  
* **강사용 강의 대본 스크립트**:"챗봇이 틀린 대답을 내놓았을 때 모델 탓만 해서는 발전이 없습니다. 'Logs' 탭을 누르면 챗봇이 그 말을 내뱉기 직전에 어떤 청크를 읽었는지 명백하게 확인할 수 있습니다14. 읽어온 청크 자체에 정답이 없었다면 '청킹과 검색 설정'의 문제이고, 청크에 정답이 버젓이 있는데도 헛소리를 했다면 '프롬프트 제약 조건'의 문제입니다14. 이 둘을 분리해 낼 수 있다면 여러분은 이미 주니어 AI 엔지니어의 통찰력을 갖춘 것입니다."

## **현장 운영 팁 및 수강생 체크리스트**

### **인덱싱 지연 및 런타임 오류 대응 트러블슈팅 5선**

| 발생 오류 및 현상 | 주요 원인 분석 | 현장 조치 프로토콜 |
| :---- | :---- | :---- |
| **1\. 문서 업로드 후 인덱싱(Indexing) 상태 영구 정지** | Celery Worker 큐 대기열 병목, Redis 통신 일시 지연, 또는 특정 특수문자에 의한 파서 무한 루프34 | ① 브라우저를 새로고침하여 상태 갱신 여부를 확인한다36. ② Self-hosted 환경의 경우 터미널에서 docker compose restart worker를 즉각 실행한다35. ③ 클라우드 환경에서는 해당 문서를 삭제하고, 서식을 제거한 순수 텍스트(TXT)로 변환해 재업로드한다35. |
| **2\. Provider Quota Exceeded 및 Not Initialize 에러** | LLM 또는 임베딩 모델의 API 크레딧 소진, 키 권한 미부여, 로컬 Ollama 포트 바인딩 단절35 | ① Dify 우측 상단 Settings ![][image1] Model Provider 진입1. ② 설정된 API Key의 유효성 검증(Validation)을 재수행. ③ Ollama 로컬 환경 사용자는 호스트를 0.0.0.0으로 바인딩하고 방화벽 11434 포트 개방 상태를 확인한다38. |
| **3\. 문서를 등록했음에도 챗봇이 "답을 모른다"고만 회피함** | 유사도 점수 하한선(Score Threshold) 과다 설정, 또는 질문과 문서 간의 어휘 불일치(Lexical Mismatch)15 | ① Retrieval Settings에서 Score Threshold를 비활성화하거나 0.4 이하로 하향 조정한다29. ② 검색 모드를 'Vector Search' 단독에서 키워드 매칭이 보완되는 'Hybrid Search'로 전환한다1. ③ Knowledge 메뉴의 Hit Testing 탭에서 질문을 직접 입력해 청크 매칭 여부를 검증한다11. |
| **4\. 지식 베이스를 무시하고 모델 마음대로 지어내어 답변함** | 시스템 프롬프트 제약 미약, 높은 모델 Temperature(창의성), 컨텍스트 경계 태그 누락4 | ① 프롬프트 내에 \<context\>{{\#context\#}}\</context\> XML 블록을 명시적으로 재배치한다26. ② LLM의 Temperature 파라미터를 0.0\~0.2 수준으로 극단적으로 하향한다. ③ "제공된 지식 외에는 어떠한 지식도 참조하지 말 것"이라는 부작용 차단 문구를 프롬프트 상·하단에 이중 배치한다2. |
| **5\. PDF 파일 업로드 시 텍스트 파싱 오류 및 글자 깨짐** | 복잡한 다단 레이아웃, 스캔 이미지 기반 PDF(OCR 미적용), 비표준 폰트 인코딩2 | ① PDF 원본 본문을 복사하여 일반 메모장에 붙여넣었을 때 정상 텍스트 레이어가 존재하는지 확인한다. ② 텍스트가 깨지거나 표가 뒤섞인 경우, Word나 Markdown(.md) 문서로 변환하여 업로드한다8. |

### **수강생 챗봇 외부 배포 단계별 안내**

챗봇 제작이 완료된 후 사용자가 이를 외부 환경에 연동하는 방식은 크게 세 가지로 나뉜다30.

* 첫째는 Dify가 호스팅하는 독립형 웹페이지 링크(Shareable Link)를 전달하는 방식이다30.  
* 둘째는 사내 인트라넷이나 기술 블로그 본문에 대화창을 인라인으로 고정하는 iframe 삽입 방식이다30.  
* 셋째는 일반 고객센터 포털처럼 화면 우측 하단에 상시 떠 있는 플로팅 챗 버블 위젯 방식이다30.

배포 방식별 특성 비교:  
\- 독립형 웹 앱 URL: 별도 사이트 구축 없이 즉시 브라우저 전체 화면으로 대화 가능 (사내 배포, 메신저 공유에 최적)  
\- 플로팅 챗 버블 위젯: 기존 웹사이트 우측 하단에 말풍선 아이콘 형태로 상시 상주 (고객지원 CS 포털에 최적)  
\- 인라인 Iframe 삽입: 웹페이지 본문 영역 한가운데 직사각형 창 형태로 내장 (노션 문서, 사내 인트라넷에 최적)

#### **1\. 독립형 웹 애플리케이션 링크 공유 (Shareable URL)**

메신저(슬랙, 카카오톡)나 모바일 환경에서 별도 사이트 연동 없이 즉각적으로 동작을 시연하고자 할 때 활용한다30.

> 1. 챗봇 스튜디오 우측 상단의 Publish(게시)를 누르고 Update를 반영한다4.  
> 2. 상단 네비게이션의 Overview(개요) 탭으로 진입한다.  
> 3. Web App 카드에 표기된 URL(예: https\://udify.app/chat/xxxxx)의 토글 스위치가 ON으로 활성화되었는지 확인한다30.  
> 4. Copy Link를 눌러 주소를 복사한 뒤 동료 및 스마트폰 브라우저에 붙여넣어 실행한다30.

#### **2\. 웹사이트 인라인 iframe 내장 (Iframe Embed)**

노션(Notion) 지식 서재나 사내 위키 본문 영역 한가운데에 고정된 검색창 형태로 챗봇을 내장하고자 할 때 활용한다30.

> 1. Overview 탭의 Web App 카드 하단에서 Embed Into Site 버튼을 클릭한다30.  
> 2. 팝업 모달에서 iframe 탭을 선택한다30.  
> 3. 제공되는 표준 HTML 코드를 복사하여 타겟 웹페이지의 원하는 레이아웃 영역에 삽입한다30.

HTML  
\<\!-- Dify Iframe 임베드 표준 코드 \--\>  
\<iframe  
  src\="https\://udify.app/chatbot/YOUR\_APP\_TOKEN"  
  style\="width: 100%; height: 700px; min-height: 600px; border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"  
  frameborder\="0"  
  allow\="microphone"\>  
\</iframe\>

#### **3\. 웹사이트 우측 하단 플로팅 챗 버블 위젯 (Floating Chat Widget)**

일반 기업 CS 웹사이트처럼 방문자가 페이지를 탐색하는 동안 우측 하단에 떠 있다가 클릭 시 열리는 대화창을 구현할 때 활용한다30.

> 1. Embed Into Site 화면에서 Chat Bubble 탭을 선택한다30.  
> 2. 타겟 웹페이지 HTML 소스 코드의 \<body\> 닫는 태그 직전에 스크립트를 삽입한다30.

HTML  
\<\!-- Dify 플로팅 챗 버블 스크립트 코드 \--\>  
\<script\>  
 window.difyChatbotConfig \= {  
  token: 'YOUR\_APP\_TOKEN'  
 }  
\</script\>  
\<script  
 src\="https\://udify.app/embed.js"  
 id\="YOUR\_APP\_TOKEN"  
 defer\>  
\</script\>

### **수강생 자가 진단 및 빌드 완료 체크리스트**

수강생이 실습을 마치고 결과물을 제출하거나 실제 서비스로 이관하기 전 점검해야 할 7가지 품질 보증 항목이다.

| 검증 영역 | 자가 진단 평가 항목 | 검증 기준 (Pass 조건) | 자가 확인 |
| :---- | :---- | :---- | :---- |
| **지식 데이터** | ① 원본 전처리 및 청킹 상태 | 특수문자 깨짐이 없으며, 개별 청크가 문맥을 온전히 유지하는가?1 | ![][image2] |
| **검색 엔진** | ② 문서 인덱싱 정상 완료 | Knowledge 목록에서 파일 상태가 'Available' 초록색 불로 활성화되었는가?34 | ![][image2] |
| **검색 튜닝** | ③ Hit Testing 적중 검증 | 지식 베이스 검색 테스트 탭에서 질문 시 0.65 이상의 유사도 점수로 청크가 노출되는가?11 | ![][image2] |
| **프롬프트** | ④ 컨텍스트 변수 결합 | 시스템 프롬프트에 컨텍스트 변수 주입 태그가 누락 없이 배치되었는가?5 | ![][image2] |
| **환각 방지** | ⑤ 비지식 질의 방어력 | 지식 문서에 없는 엉뚱한 질문을 던졌을 때, 지어내지 않고 정해진 거절 멘트를 출력하는가?2 | ![][image2] |
| **출처 표기** | ⑥ Citation 정상 표시 | 답변 하단에 근거 문서명과 참조 청크를 확인할 수 있는 출처 배지가 뜨는가?25 | ![][image2] |
| **배포 완료** | ⑦ 모바일 외부 접속성 | 발행된 웹 앱 URL 링크를 스마트폰으로 열어 실시간 질의응답이 오가는가?30 | ![][image2] |

## **교육 효과 극대화를 위한 종합 제언**

RAG 시스템 구축 교육에서 초심자들이 범하는 가장 흔한 오류는 'LLM이 알아서 문맥을 추론해 줄 것'이라는 막연한 신뢰에서 비롯되는 데이터 전처리 방치와 모호한 프롬프트 지시문이다2. Dify는 복잡한 인프라와 파이프라인을 노코드 환경으로 추상화하였으나, 결과물의 신뢰성은 전적으로 '얼마나 정제된 청크를 입력했는가'와 '프롬프트의 제약 조건을 얼마나 빈틈없이 설정했는가'에 의해 좌우된다14.  
따라서 교육자는 단순한 UI 기능 소개에 머물지 않고, 수강생이 의도적인 이상 질의(Out-of-domain query)를 입력하여 오답을 발생시키게 한 후 Logs 탭에서 매칭된 청크를 직접 추적해 보는 디버깅 실습을 필히 거치도록 유도해야 한다14.  
검색 매개변수(Top-K, 유사도 임계값)의 미세한 변화가 비즈니스 정밀도에 미치는 영향력을 체감할 때 비로소 수강생은 흥미 위주의 단순 챗봇을 넘어 현업 실무에 즉시 투입 가능한 엔터프라이즈급 지식 애플리케이션의 아키텍트로 거듭날 수 있다11.

#### **Works cited**

> 1. Customer Service Bot With Knowledge Base \- Dify Docs, [https\://docs.dify.ai/en/learn/tutorials/customer-service-bot](https://docs.dify.ai/en/learn/tutorials/customer-service-bot)  
> 2. How to Build an AI Chatbot with a Custom Knowledge Base, [https\://www\.fwdslash.ai/blog/how-to-build-an-ai-chatbot-with-a-custom-knowledge-base](https://www.fwdslash.ai/blog/how-to-build-an-ai-chatbot-with-a-custom-knowledge-base)  
> 3. Retrieval Augmented Generation \- Sean's Blog, [https\://seanpedersen.github.io/posts/rag/](https://seanpedersen.github.io/posts/rag/)  
> 4. Text Generator \- Dify Docs, [https\://docs.dify.ai/en/cloud/use-dify/build/text-generator](https://docs.dify.ai/en/cloud/use-dify/build/text-generator)  
> 5. Knowledge Retrieval: A Smart Chatbot \- Dify Marketplace, [https\://marketplace.dify.ai/template/langgenius/d2b08abe-0028-4459-8ef0-d2509a75c9ff](https://marketplace.dify.ai/template/langgenius/d2b08abe-0028-4459-8ef0-d2509a75c9ff)  
> 6. Flowise is shutting down in 2026 | alternatives, timeline, risks \- wonk.ai, [https\://www\.wonk.ai/en/blog/2026-07-flowise-wird-eingestellt/](https://www.wonk.ai/en/blog/2026-07-flowise-wird-eingestellt/)  
> 7. Dify | Drupal.org, [https\://www\.drupal.org/project/dify](https://www.drupal.org/project/dify)  
> 8. Introducing Knowledge Pipeline \- Dify, [https\://dify.ai/blog/introducing-knowledge-pipeline](https://dify.ai/blog/introducing-knowledge-pipeline)  
> 9. Dify Self-Hosted: Deploy Your Own AI Platform \- Local AI Master, [https\://localaimaster.com/blog/dify-self-hosted-setup-guide](https://localaimaster.com/blog/dify-self-hosted-setup-guide)  
> 10. Dify v0.15.0: New Parent-Child Retrieval Strategy \- AI分享圈, [https\://aisharenet.com/en/dify-v0150quanban/](https://aisharenet.com/en/dify-v0150quanban/)  
> 11. Knowledge Retrieval \- Dify Docs, [https\://docs.dify.ai/en/cloud/use-dify/nodes/knowledge-retrieval](https://docs.dify.ai/en/cloud/use-dify/nodes/knowledge-retrieval)  
> 12. Dify \- Social Media Glossary | Publora, [https\://publora.com/glossary/dify](https://publora.com/glossary/dify)  
> 13. Why the “Parent-Child Mode” is Your AI's New Weapon \- Medium, [https\://chwang12341.medium.com/rag-knowledge-bases-family-meeting-why-the-parent-child-mode-is-your-ai-s-new-weapon-22198b390197](https://chwang12341.medium.com/rag-knowledge-bases-family-meeting-why-the-parent-child-mode-is-your-ai-s-new-weapon-22198b390197)  
> 14. How to Build a Dify RAG Chatbot: Step-by-Step Workflow Guide, [https\://workflows.so/blog/how-to-build-a-dify-rag-chatbot-step-by-step-workflow-guide](https://workflows.so/blog/how-to-build-a-dify-rag-chatbot-step-by-step-workflow-guide)  
> 15. RAG that actually works? : r/LocalLLaMA \- Reddit, [https\://www\.reddit.com/r/LocalLLaMA/comments/1ps6txq/rag\_that\_actually\_works/](https://www.reddit.com/r/LocalLLaMA/comments/1ps6txq/rag_that_actually_works/)  
> 16. Markdown Text Chunker \- Dify Marketplace, [https\://marketplace.dify.ai/plugin/biaoo/md-text-chunker](https://marketplace.dify.ai/plugin/biaoo/md-text-chunker)  
> 17. Parent-Child Chunking: Creating Hierarchical Document Segments, [https\://www\.sandgarden.com/learn/parent-child-chunking](https://www.sandgarden.com/learn/parent-child-chunking)  
> 18. The Evolution of RAG Text Chunking: Why Precision Still Matters, [https\://tao-hpu.medium.com/the-evolution-of-rag-text-chunking-why-precision-still-matters-c3e35ef79c50](https://tao-hpu.medium.com/the-evolution-of-rag-text-chunking-why-precision-still-matters-c3e35ef79c50)  
> 19. Dynamic Knowledge Retrieval \- Dify Marketplace, [https\://marketplace.dify.ai/plugin/xiaozie-xy/dynamic\_knowledge](https://marketplace.dify.ai/plugin/xiaozie-xy/dynamic_knowledge)  
> 20. \[Dify Hands-on Verification\] 3 Reasons RAG Accuracy Stagnates, [https\://note.com/mild\_yeti5182/n/nf37fec800d25?hl=en](https://note.com/mild_yeti5182/n/nf37fec800d25?hl=en)  
> 21. 从Dify 配置页理解RAG 的重要参数原创 \- CSDN博客, [https\://blog.csdn.net/hongke\_weixin/article/details/161286669](https://blog.csdn.net/hongke_weixin/article/details/161286669)  
> 22. \[Dify\] 지식 생성, [https\://hardworkingini.tistory.com/191](https://hardworkingini.tistory.com/191)  
> 23. Knowledge base retrieval efficiency problem · Issue \#13553, [https\://github.com/langgenius/dify/issues/13553](https://github.com/langgenius/dify/issues/13553)  
> 24. Dify RAG No-Hit Response: Prompt Control or IF-ELSE ... \- Zenn, [https\://zenn.dev/okzaq/articles/dify-rag-nohit-pattern?locale=en](https://zenn.dev/okzaq/articles/dify-rag-nohit-pattern?locale=en)  
> 25. LLM \- Dify Docs, [https\://docs.dify.ai/en/cloud/use-dify/nodes/llm](https://docs.dify.ai/en/cloud/use-dify/nodes/llm)  
> 26. Enhance Dify RAG with InfraNodus: Expand Your LLM's Context, [https\://dify.ai/blog/enhance-dify-rag-with-infranodus-expand-your-llm-s-context](https://dify.ai/blog/enhance-dify-rag-with-infranodus-expand-your-llm-s-context)  
> 27. how to convert the knowledge content return from '\\r\\n' to '\\n' · Issue, [https\://github.com/langgenius/dify/issues/19576](https://github.com/langgenius/dify/issues/19576)  
> 28. Awesome list of Retrieval-Augmented Generation (RAG ... \- GitHub, [https\://github.com/Danielskry/Awesome-RAG](https://github.com/Danielskry/Awesome-RAG)  
> 29. Dify Integration | relaxAI Docs, [https\://relax.ai/docs/integrations/agentic-ai-platforms/dify](https://relax.ai/docs/integrations/agentic-ai-platforms/dify)  
> 30. Embed Your Web App \- Dify Docs, [https\://docs.dify.ai/en/cloud/use-dify/publish/webapp/embedding-in-websites](https://docs.dify.ai/en/cloud/use-dify/publish/webapp/embedding-in-websites)  
> 31. Help:how should I configure it so that the agent can automatically, [https\://github.com/langgenius/dify/discussions/30266](https://github.com/langgenius/dify/discussions/30266)  
> 32. User Input \- Dify Docs, [https\://docs.dify.ai/en/self-host/use-dify/nodes/user-input](https://docs.dify.ai/en/self-host/use-dify/nodes/user-input)  
> 33. RAG \- AgentScope Java, [https\://java.agentscope.io/v1/en/docs/task/rag](https://java.agentscope.io/v1/en/docs/task/rag)  
> 34. RAG stuck in embedding with status 'indexing' when using ... \- GitHub, [https\://github.com/langgenius/dify/issues/4564](https://github.com/langgenius/dify/issues/4564)  
> 35. upload documents hang INDEXING. \#15434 \- langgenius/dify \- GitHub, [https\://github.com/langgenius/dify/issues/15434](https://github.com/langgenius/dify/issues/15434)  
> 36. Files Stuck in Queuing/Indexing State in Dify Knowledge Base \#30154, [https\://github.com/langgenius/dify/issues/30154](https://github.com/langgenius/dify/issues/30154)  
> 37. Weaviate Migration Guide upgrading to Client v4 and Server 1.27+, [https\://docs.dify.ai/en/self-host/deploy/troubleshooting/weaviate-v4-migration](https://docs.dify.ai/en/self-host/deploy/troubleshooting/weaviate-v4-migration)  
> 38. EmbeddingGemma indexing is extremely slow when used for, [https\://discuss.ai.google.dev/t/embeddinggemma-indexing-is-extremely-slow-when-used-for-knowledge-indexing-in-dify-docker-setup/116374](https://discuss.ai.google.dev/t/embeddinggemma-indexing-is-extremely-slow-when-used-for-knowledge-indexing-in-dify-docker-setup/116374)  
> 39. 12 Best Open-Source AI Agent Frameworks on GitHub (2026), [https\://www\.ayautomate.com/blog/best-open-source-ai-agent-frameworks](https://www.ayautomate.com/blog/best-open-source-ai-agent-frameworks)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAATCAYAAACdkl3yAAAAZUlEQVR4XmNgGAWjYIgDEyDeiC5IDuAA4q3oguSCICCWRRckFzwC4hx0QW8gDiEDdwPxKSBmZoCCuwwQG0jFz4D4PxDrMlAAZgHxIXRBcsB1IJZHFyQVNAAxK7ogqcAUiG+gCwIA4iQblKUwv34AAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAATCAYAAACgADyUAAAAM0lEQVR4XmNgGAW0A/+B+BGR+AkQ+0K0MTA8hDGIADwMoxpxglGNeACKxr8MkPRKLPYAAMJoJL0M72J6AAAAAElFTkSuQmCC>