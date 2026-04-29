import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from typing import Any
from langchain_perplexity import ChatPerplexity
from datetime import datetime
import logging
import re
import base64

# 환경 변수 로드
load_dotenv()

# 로깅 설정
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"hdoilbank_chatbot_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# HTTP 요청 로그 비활성화
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# 구분선 및 취소선 제거 함수
def remove_separators(text: str) -> str:
    """답변에서 구분선(---, ===, ___)과 취소선(~~텍스트~~)을 제거합니다."""
    if not text:
        return text
    # 취소선 마크다운 제거 (~~텍스트~~ -> 텍스트)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # 여러 줄에 걸친 구분선 제거 (공백 포함)
    text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*={3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*_{3,}\s*\n', '\n\n', text)
    # 단독 라인의 구분선 제거
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*={3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}\s*$', '', text, flags=re.MULTILINE)
    # 연속된 빈 줄 정리 (최대 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# LLM 모델 선택 함수
def get_llm(model_name: str, temperature: float = 0.7) -> Any:
    """선택된 모델명에 따라 적절한 LLM 인스턴스를 반환합니다."""
    if model_name == "gpt-5.1":
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)
    elif model_name == "claude-sonnet-4-5":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
    elif model_name == "gemini-3-pro-preview":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY가 환경변수에 설정되어 있지 않습니다.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=api_key, temperature=temperature)
    else:
        # 기본값: gpt-5.1
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)

# 페이지 설정
st.set_page_config(
    page_title="HD현대오일뱅크 챗봇",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 초기 상태 설정
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "search_model" not in st.session_state:
    st.session_state.search_model = "사용 안 함"

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-5.1"

# CSS 스타일 (HD현대오일뱅크 톤: 그룹 남청·에너지 블루 포인트)
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

    html, body, [class*="css"] {
        font-family: 'Pretendard', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(165deg, #f4f7fb 0%, #e9f0f7 50%, #e2ebf4 100%) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.97) !important;
        border-bottom: 1px solid #b8cddd !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafcfe 0%, #f2f7fb 100%) !important;
        border-right: 1px solid #c5d8e8 !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
    }

    .main .block-container {
        padding-top: 1.5rem !important;
        max-width: 1200px !important;
    }

    /* 헤딩 — HD 계열 남청 */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #062843 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    .main h1 { font-size: 1.35rem !important; }
    .main h2 { font-size: 1.15rem !important; color: #005a9c !important; }
    .main h3 { font-size: 1.05rem !important; }
    .main h4, .main h5, .main h6 { font-size: 1rem !important; }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #005a9c !important;
        font-weight: 700 !important;
    }

    [data-testid="stChatMessageContainer"] {
        background: #ffffff !important;
        border: 1px solid #c9dae8 !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(0, 91, 150, 0.06) !important;
        margin-bottom: 0.65rem !important;
    }

    .stChatMessage {
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        color: #1c2833 !important;
    }

    .stChatMessage p {
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        margin: 0.5rem 0 !important;
    }

    .stChatMessage ul, .stChatMessage ol {
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        margin: 0.5rem 0 !important;
        padding-left: 1.25rem !important;
    }

    .stChatMessage li {
        margin: 0.35rem 0 !important;
    }

    .stChatMessage strong, .stChatMessage b {
        font-weight: 700 !important;
        color: #062843 !important;
    }

    .stChatMessage blockquote {
        margin: 0.5rem 0 !important;
        padding: 0.5rem 0.85rem !important;
        border-left: 4px solid #00a4e3 !important;
        background: #f0f7fc !important;
        border-radius: 0 6px 6px 0 !important;
    }

    .stChatMessage code {
        font-size: 0.88rem !important;
        background-color: #edf4fa !important;
        color: #062843 !important;
        padding: 0.15rem 0.4rem !important;
        border-radius: 4px !important;
        border: 1px solid #c9dae8 !important;
    }

    .stChatMessage * {
        font-family: 'Pretendard', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif !important;
    }

    [data-testid="stChatInput"] {
        border-top: 1px solid #b8cddd !important;
        background: #f5f9fc !important;
    }
    [data-testid="stChatInput"] textarea {
        border: 1px solid #9ebad0 !important;
        border-radius: 6px !important;
    }

    .stButton > button {
        background-color: #005a9c !important;
        color: #ffffff !important;
        border: 1px solid #003b73 !important;
        border-radius: 6px !important;
        padding: 0.45rem 1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }
    .stButton > button:hover {
        background-color: #003b73 !important;
        border-color: #002a52 !important;
    }

    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        font-size: 0.9rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        border: 1px dashed #7aa3c4 !important;
        border-radius: 6px !important;
        background: #ffffff !important;
    }

    p.hdo-side-h {
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        color: #003b73 !important;
        margin: 1.1rem 0 0.4rem 0 !important;
        padding-bottom: 0.35rem !important;
        border-bottom: 2px solid #00a4e3 !important;
        letter-spacing: -0.02em !important;
    }
    p.hdo-side-h:first-child { margin-top: 0 !important; }
    p.hdo-side-h3 {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #2d4d66 !important;
        margin: 0.85rem 0 0.35rem 0 !important;
    }

    .hdo-page-header {
        background: #ffffff;
        border: 1px solid #b8cddd;
        border-top: 4px solid #00a4e3;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0, 59, 115, 0.08);
    }
    .hdo-page-header-inner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.25rem 1.75rem;
        flex-wrap: wrap;
    }
    .hdo-page-header-text {
        flex: 1 1 280px;
        min-width: 0;
    }
    .hdo-page-header-logo {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-left: 0.5rem;
    }
    .hdo-page-header-logo img {
        display: block;
        max-height: 76px;
        width: auto;
        max-width: min(260px, 40vw);
        object-fit: contain;
    }
    .hdo-page-header .hdo-en {
        font-size: 0.72rem;
        color: #5a6d80;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .hdo-page-header h1.hdo-title,
    .hdo-page-header .hdo-title {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #003b73 !important;
        margin: 0 !important;
        line-height: 1.3 !important;
        letter-spacing: -0.03em !important;
    }
    .hdo-page-header .hdo-title span.sub {
        font-weight: 600;
        color: #005a9c;
    }
    .hdo-page-header .hdo-desc {
        font-size: 0.9rem;
        color: #3d556b;
        margin-top: 0.5rem;
        line-height: 1.55;
    }

    .hdo-notice-bar {
        font-size: 0.82rem;
        color: #2d4d66;
        background: linear-gradient(90deg, #e8f4fc 0%, #eef6fb 100%);
        border: 1px solid #9ec5e0;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

def _mime_for_logo_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        return "image/svg+xml"
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


# 로고 및 제목 영역 (텍스트 좌측 · 로고 우측, 한 블록으로 정렬)
_logo_file = "logo.svg"
_logo_paths = [
    "hd_oilbank.svg",
    "HD_Hyundai_Oilbank.svg",
    "logo.svg",
    "logo.png",
    "logo.jpg",
    "assets/logo.svg",
    "assets/logo.png",
    "images/logo.svg",
    "images/logo.png",
    "static/logo.svg",
    "static/logo.png",
]

_logo_inner = ""
for _path in _logo_paths:
    if not os.path.exists(_path):
        continue
    try:
        with open(_path, "rb") as _lf:
            _b64 = base64.standard_b64encode(_lf.read()).decode("ascii")
        _mime = _mime_for_logo_path(_path)
        _logo_inner = (
            f'<img src="data:{_mime};base64,{_b64}" alt="HD현대오일뱅크" '
            'loading="lazy" decoding="async" />'
        )
        break
    except Exception as _e:
        logger.warning("로고 파일 로드 실패 (%s): %s", _path, _e)

if not _logo_inner:
    # 한 줄로 유지: 여러 줄+들여쓰기 HTML은 st.markdown이 마크다운 '코드 블록'으로 해석해 그대로 노출될 수 있음
    _logo_inner = (
        '<div style="min-width:160px;max-width:240px;height:64px;display:flex;align-items:center;'
        'justify-content:center;padding:0 12px;background:linear-gradient(135deg,#003b73 0%,#005a9c 100%);'
        'border:1px solid #002a52;border-radius:6px;">'
        '<span style="color:#fff;font-size:0.72rem;font-weight:700;letter-spacing:0.04em;text-align:center;">'
        "HD HYUNDAI OILBANK</span></div>"
    )
    logger.warning("로고 파일을 찾을 수 없습니다: %s", _logo_file)

st.markdown(
    f"""<div class="hdo-page-header"><div class="hdo-page-header-inner"><div class="hdo-page-header-text">
<div class="hdo-en">HD Hyundai Oilbank</div>
<h1 class="hdo-title">HD현대오일뱅크 <span class="sub">챗봇</span></h1>
<p class="hdo-desc">질의 응답과 문서(PDF) 기반 검색을 함께 사용할 수 있습니다. 좌측 패널에서 언어 모델·인터넷 검색·RAG(PDF) 사용 여부를 설정한 뒤 하단에 질문해 주십시오.</p>
</div><div class="hdo-page-header-logo">{_logo_inner}</div></div></div>""",
    unsafe_allow_html=True,
)

st.markdown("""
<div class="hdo-notice-bar">
    <strong>이용 안내</strong> — 좌측 패널에서 LLM 모델, 인터넷 검색(Perplexity), RAG(PDF) 옵션을 선택한 뒤 하단 입력란에 질문을 입력해 주십시오.
</div>
""", unsafe_allow_html=True)

# Perplexity 검색 함수
def search_with_perplexity_chat(prompt: str, history: list, temperature: float = 0.7) -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "PERPLEXITY_API_KEY가 환경변수에 설정되어 있지 않습니다."
    
    messages = []
    system_content = """당신은 전문적인 AI 어시스턴트입니다. 답변을 전문적으로 해줘.

답변 형식:
- 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
- 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
- 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
- 답변은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
- 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요"""
    messages.append({"role": "system", "content": system_content})

    filtered_history = []
    last_role = "system"
    for msg in history:
        if msg["role"] not in ["user", "assistant"]:
            continue
        if last_role == "system" and msg["role"] != "user":
            continue
        if last_role == msg["role"]:
            continue
        filtered_history.append(msg)
        last_role = msg["role"]

    messages.extend(filtered_history)

    if not filtered_history or filtered_history[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": prompt})

    llm = ChatPerplexity(model="sonar-pro", temperature=temperature, api_key=api_key)
    try:
        response = llm.invoke(messages)
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    except Exception as e:
        return f"Perplexity 검색 중 오류: {e}"

# 사이드바 설정
with st.sidebar:
    # 1. LLM 모델 선택
    st.markdown('<p class="hdo-side-h">1. LLM 모델 선택</p>', unsafe_allow_html=True)
    all_models = ["gpt-5.1", "gemini-3-pro-preview", "claude-sonnet-4-5"]
    
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = all_models[0]
    
    try:
        current_index = all_models.index(st.session_state.llm_model)
    except ValueError:
        current_index = 0
    
    selected_model = st.radio(
        "사용할 언어모델을 선택하세요",
        options=all_models,
        index=current_index,
        key="llm_model_radio_hdoilbank",
    )
    st.session_state.llm_model = selected_model

    # 2. 인터넷 검색 선택
    st.markdown('<p class="hdo-side-h">2. 인터넷 검색</p>', unsafe_allow_html=True)
    search_model = st.radio(
        "인터넷 검색을 사용하시겠습니까?",
        [
            "사용 안 함",
            "Perplexity 사용"
        ],
        index=0 if st.session_state.search_model == "사용 안 함" else 1
    )
    st.session_state.search_model = search_model

    # 3. RAG 선택
    st.markdown('<p class="hdo-side-h">3. RAG (PDF 검색)</p>', unsafe_allow_html=True)
    use_rag = st.radio(
        "RAG를 사용하시겠습니까?",
        [
            "사용 안 함",
            "RAG 사용"
        ],
        index=0 if not st.session_state.use_rag else 1
    )
    st.session_state.use_rag = (use_rag == "RAG 사용")

    # 4. PDF 파일 업로드
    st.markdown('<p class="hdo-side-h">4. PDF 파일 업로드</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        process_button = st.button("파일 처리하기")
        
        if process_button:
            with st.spinner("PDF 파일을 처리 중입니다..."):
                try:
                    # 임시 파일 생성 및 처리
                    temp_dir = tempfile.TemporaryDirectory()
                    
                    all_docs = []
                    new_files = []
                    
                    # 각 파일 처리
                    for uploaded_file in uploaded_files:
                        # 이미 처리된 파일 스킵
                        if uploaded_file.name in st.session_state.processed_files:
                            continue
                            
                        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                        
                        # 업로드된 파일을 임시 파일로 저장
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # PDF 로더 생성 및 문서 로드
                        loader = PyPDFLoader(temp_file_path)
                        documents = loader.load()
                        
                        # 메타데이터에 파일 이름 추가
                        for doc in documents:
                            doc.metadata["source"] = uploaded_file.name
                        
                        all_docs.extend(documents)
                        new_files.append(uploaded_file.name)
                
                    if not all_docs:
                        st.success("모든 파일이 이미 처리되었습니다.")
                    else:
                        # 텍스트 분할
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=100,
                            length_function=len
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        
                        # 모든 청크를 벡터 데이터베이스에 저장
                        total_chunks = len(chunks)
                        
                        # 임베딩 및 벡터 스토어 생성
                        embeddings = OpenAIEmbeddings()
                        
                        if st.session_state.vectorstore is None:
                            # 새 벡터 스토어 생성
                            batch_size = 30
                            vectorstore = None
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                
                                try:
                                    if vectorstore is None:
                                        vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                                    else:
                                        vectorstore.add_documents(batch_chunks)
                                except Exception as e:
                                    continue
                            
                            st.session_state.vectorstore = vectorstore
                        else:
                            # 기존 벡터 스토어에 추가
                            batch_size = 30
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                
                                try:
                                    st.session_state.vectorstore.add_documents(batch_chunks)
                                except Exception as e:
                                    continue
                        
                        # 검색기 생성 (더 많은 결과와 정확한 검색)
                        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 10}  # 검색 결과 수 증가
                        )
                        
                        # 처리된 파일 목록 업데이트
                        st.session_state.processed_files.extend(new_files)
                        
                except Exception as e:
                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                    st.error("파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")
                    logger.error(f"PDF 파일 처리 오류: {e}")

    # 처리된 파일 목록 표시
    if st.session_state.processed_files:
        st.markdown('<p class="hdo-side-h3">처리된 파일 목록</p>', unsafe_allow_html=True)
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.rerun()
    
    # 현재 설정 표시
    st.markdown('<p class="hdo-side-h3">현재 설정</p>', unsafe_allow_html=True)
    st.text(f"모델: {st.session_state.llm_model}")
    st.text(f"인터넷 검색: {st.session_state.search_model}")
    st.text(f"RAG: {'사용' if st.session_state.use_rag else '사용 안 함'}")
    if st.session_state.processed_files:
        st.text(f"처리된 파일: {len(st.session_state.processed_files)}개")
        st.text(f"대화 기록: {len(st.session_state.chat_history)}개")

# 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            st.write(message["content"])

# 사용자 입력 영역
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # --- 1순위: 인터넷 검색 모델 처리 ---
    if st.session_state.search_model == "Perplexity 사용":
        with st.spinner("Perplexity 검색 중..."):
            chat_history_for_perplexity = []
            for msg in st.session_state.chat_history[:-1]:
                if msg["role"] in ["user", "assistant"]:
                    chat_history_for_perplexity.append({"role": msg["role"], "content": msg["content"]})

            response_text = ""
            try:
                response_text = search_with_perplexity_chat(prompt, chat_history_for_perplexity)
                if not response_text or not isinstance(response_text, str):
                    response_text = "Perplexity 검색 결과가 없습니다."
                
                # 답변에서 구분선 제거
                response_text = remove_separators(response_text)
                
                # 다음 질문 3개 생성
                try:
                    llm = get_llm(st.session_state.llm_model, temperature=1)
                    next_questions_prompt = f"""
                    질문자가 한 질문: {prompt}
                    
                    생성된 답변:
                    {response_text}
                    
                    위 질문과 답변 내용을 검토하여, 질문자가 다음에 할 수 있는 중요한 3가지 질문을 생성해주세요.
                    
                    요구사항:
                    - 답변 내용을 더 깊이 이해하기 위한 후속 질문
                    - 답변에서 언급된 내용을 구체화하거나 확장하는 질문
                    - 관련된 다른 주제나 관점을 탐색할 수 있는 질문
                    - 각 질문은 완전한 문장으로 작성하되, 간결하고 명확하게 작성
                    - 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성
                    
                    형식:
                    질문1
                    질문2
                    질문3
                    
                    참고: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요.
                    """
                    next_questions_response = llm.invoke(next_questions_prompt).content
                    next_questions = [q.strip() for q in next_questions_response.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                    next_questions = next_questions[:3]
                    
                    if next_questions:
                        response_text += "\n\n"
                        response_text += "### 관련 질문 안내\n\n"
                        for i, question in enumerate(next_questions, 1):
                            response_text += f"{i}. {question}\n\n"
                except Exception as e:
                    logger.warning(f"다음 질문 생성 실패: {e}")

                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_msg = f"Perplexity 검색 중 오류: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                logger.error(f"Perplexity 검색 오류: {e}")

    # --- 2순위: RAG 또는 직접 LLM 답변 (인터넷 검색 '사용 안 함'일 때만 실행) ---
    elif st.session_state.search_model == "사용 안 함":
        # --- 2-1: RAG 사용이 선택되었고 PDF 파일이 있는 경우 ---
        if st.session_state.use_rag and st.session_state.retriever is not None:
            with st.spinner("PDF 기반 RAG 답변을 생성 중입니다..."):
                try:
                    # RAG 검색 (상위 3개 문서만 사용)
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    
                    if not retrieved_docs:
                        response = f"죄송합니다. '{prompt}'에 대한 관련 문서를 찾을 수 없습니다."
                    else:
                        # 상위 3개 문서만 사용
                        top_docs = retrieved_docs[:3]
                        
                        # 컨텍스트 구성
                        context_text = ""
                        max_context_length = 8000
                        current_length = 0
                        
                        for i, doc in enumerate(top_docs):
                            doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                            if current_length + len(doc_text) > max_context_length:
                                st.warning(f"토큰 제한으로 인해 문서 {i+1}개만 사용합니다.")
                                break
                            context_text += doc_text
                            current_length += len(doc_text)
                        
                        # 과거 대화 맥락 구성
                        conversation_context = ""
                        if st.session_state.conversation_memory:
                            conversation_context = "\n\n=== 이전 대화 맥락 ===\n"
                            # 최근 50개 대화 사용
                            recent_conversations = st.session_state.conversation_memory[-50:]
                            for conv in recent_conversations:
                                conversation_context += f"{conv}\n"
                            conversation_context += "=== 대화 맥락 끝 ===\n"
                        
                        # 시스템 프롬프트 구성
                        system_prompt = f"""
                        질문: {prompt}
                        
                        관련 문서:
                        {context_text}{conversation_context}
                        
                        위 문서 내용과 이전 대화 맥락을 모두 고려하여 질문에 답변해주세요.
                        이전 대화에서 언급된 내용이 있다면 그것을 참조하여 더 정확하고 맥락적인 답변을 제공하세요.
                        
                        답변 형식:
                        - 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
                        - 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
                        - 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
                        - 답변은 서술형으로 작성하되 존대말을 사용하세요
                        - 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요
                        
                        주의사항:
                        - 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
                        - "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
                        - 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
                        - 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
                        - 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
                        - 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
                        - 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
                        - 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요
                        """
                        
                        # LLM으로 답변 생성 (스트리밍 모드)
                        llm = get_llm(st.session_state.llm_model, temperature=1)
                        
                        response = ""
                        with st.chat_message("assistant"):
                            stream_placeholder = st.empty()
                            # 스트리밍으로 답변 생성
                            for chunk in llm.stream(system_prompt):
                                if hasattr(chunk, 'content'):
                                    chunk_text = chunk.content
                                else:
                                    chunk_text = str(chunk)
                                response += chunk_text
                                # 실시간으로 표시 (구분선 제거 포함)
                                cleaned_response = remove_separators(response)
                                stream_placeholder.markdown(cleaned_response)
                        
                        # 답변에서 구분선 제거
                        response = remove_separators(response)
                    
                        # 다음 질문 3개 생성
                        next_questions_prompt = f"""
                        질문자가 한 질문: {prompt}
                        
                        생성된 답변:
                        {response}
                        
                        위 질문과 답변 내용을 검토하여, 질문자가 다음에 할 수 있는 중요한 3가지 질문을 생성해주세요.
                        
                        요구사항:
                        - 답변 내용을 더 깊이 이해하기 위한 후속 질문
                        - 답변에서 언급된 내용을 구체화하거나 확장하는 질문
                        - 관련된 다른 주제나 관점을 탐색할 수 있는 질문
                        - 각 질문은 완전한 문장으로 작성하되, 간결하고 명확하게 작성
                        - 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성
                        
                        형식:
                        질문1
                        질문2
                        질문3
                        
                        참고: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요.
                        """
                        
                        try:
                            next_questions_response = llm.invoke(next_questions_prompt).content
                            # 질문들을 리스트로 파싱
                            next_questions = [q.strip() for q in next_questions_response.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                            # 최대 3개만 선택
                            next_questions = next_questions[:3]
                            
                            # 답변 끝에 다음 질문 추가
                            if next_questions:
                                response += "\n\n"
                                response += "### 관련 질문 안내\n\n"
                                for i, question in enumerate(next_questions, 1):
                                    response += f"{i}. {question}\n\n"
                                # 다음 질문 추가 후 다시 표시
                                with st.chat_message("assistant"):
                                    st.markdown(response)
                        except Exception as e:
                            # 다음 질문 생성 실패 시 무시하고 원래 답변만 표시
                            logger.warning(f"다음 질문 생성 실패: {e}")
                        
                        # 대화 기록에 추가
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # 대화 맥락 메모리에 추가 (최근 50개 대화 유지)
                        st.session_state.conversation_memory.append(f"사용자: {prompt}")
                        st.session_state.conversation_memory.append(f"AI: {response}")
                        if len(st.session_state.conversation_memory) > 100:  # 50개 대화 = 100개 메시지
                            st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                    
                except Exception as e:
                    with st.chat_message("assistant"):
                        st.write(f"오류가 발생했습니다: {str(e)}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
                    logger.error(f"RAG 답변 생성 오류: {e}")

        # --- 2-2: RAG 사용이 선택되지 않았거나 PDF 파일이 없는 경우 (직접 LLM 사용) ---
        else:
            if st.session_state.use_rag and st.session_state.retriever is None:
                with st.chat_message("assistant"):
                    st.warning("RAG를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요.")
                st.session_state.chat_history.append({"role": "assistant", "content": "RAG를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요."})
                logger.warning("RAG 선택되었으나 PDF 파일이 없음")
            else:
                try:
                    llm = get_llm(st.session_state.llm_model, temperature=1)
                    direct_prompt = f"""당신은 유능한 AI 어시스턴트입니다. 반드시 한국어로 답변해주세요.

질문: {prompt}

답변 형식:
- 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
- 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
- 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
- 답변은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
- 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요"""
                    
                    response = ""
                    with st.chat_message("assistant"):
                        stream_placeholder = st.empty()
                        # 스트리밍으로 답변 생성
                        for chunk in llm.stream(direct_prompt):
                            if hasattr(chunk, 'content'):
                                chunk_text = chunk.content
                            else:
                                chunk_text = str(chunk)
                            response += chunk_text
                            # 실시간으로 표시 (구분선 제거 포함)
                            cleaned_response = remove_separators(response)
                            stream_placeholder.markdown(cleaned_response)
                    
                    # 답변에서 구분선 제거
                    response = remove_separators(response)
                    
                    # 다음 질문 3개 생성
                    try:
                        next_questions_prompt = f"""
                        질문자가 한 질문: {prompt}
                        
                        생성된 답변:
                        {response}
                        
                        위 질문과 답변 내용을 검토하여, 질문자가 다음에 할 수 있는 중요한 3가지 질문을 생성해주세요.
                        
                        요구사항:
                        - 답변 내용을 더 깊이 이해하기 위한 후속 질문
                        - 답변에서 언급된 내용을 구체화하거나 확장하는 질문
                        - 관련된 다른 주제나 관점을 탐색할 수 있는 질문
                        - 각 질문은 완전한 문장으로 작성하되, 간결하고 명확하게 작성
                        - 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성
                        
                        형식:
                        질문1
                        질문2
                        질문3
                        
                        참고: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요.
                        """
                        next_questions_response = llm.invoke(next_questions_prompt).content
                        next_questions = [q.strip() for q in next_questions_response.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                        next_questions = next_questions[:3]
                        
                        if next_questions:
                            response += "\n\n"
                            response += "### 관련 질문 안내\n\n"
                            for i, question in enumerate(next_questions, 1):
                                response += f"{i}. {question}\n\n"
                            # 다음 질문 추가 후 다시 표시
                            with st.chat_message("assistant"):
                                st.markdown(response)
                    except Exception as e:
                        logger.warning(f"다음 질문 생성 실패: {e}")
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"LLM 생성 중 오류 발생: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    logger.error(f"LLM 답변 생성 오류: {e}")