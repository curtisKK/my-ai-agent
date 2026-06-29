import os
import datetime
import yfinance as yf
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
# 💡 [수정 1] DuckDuckGo 대신 Tavily를 가져옵니다.
from langchain_community.tools.tavily_search import TavilySearchResults 

# 1. 웹사이트 UI 설정 
st.set_page_config(page_title="주식 AI 에이전트", page_icon="📈", layout="centered")
st.title("📈 주식 분석 AI 에이전트 By 강재원 ")
st.markdown("2026년 7월 SK신입구성원교육 과제 입니다! ")

# 🚨 2. API 키 보안 설정
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    # 💡 [수정 2] Tavily API 키도 금고에서 가져와 환경변수에 등록합니다.
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"] 
except KeyError:
    st.error("API 키(Google 또는 Tavily)가 금고(secrets.toml)에 설정되지 않았습니다.")
    st.stop()

# 3. 도구(Tools) 정의
@tool
def multiply(a: float, b: float) -> float:
    """두 숫자를 곱하는 계산기 도구입니다."""
    return a * b

@tool
def calculate_average(a: float, b: float) -> float:
    """두 숫자의 평균값을 계산하는 도구입니다."""
    return (a + b) / 2

@tool
def get_today_date() -> str:
    """오늘의 날짜를 반환하는 도구입니다."""
    return datetime.datetime.now().strftime("%Y년 %m월 %d일")

@tool
def get_korean_stock_price(ticker: str) -> str:
    """
    특정 한국 주식 종목의 실시간 주가를 가져옵니다.
    입력값(ticker)은 반드시 '000660.KS' 같은 코드 형태여야 합니다.
    """
    # 💡 [수정] 여러 도구에서 공통으로 쓸 수 있게 딕셔너리를 밖으로 뺍니다.
TICKER_MAP = {
    "SK하이닉스": "000660.KS",
    "삼성전자": "005930.KS",
    "카카오": "035720.KS",
    "현대차": "005380.KS",
    "네이버": "035420.KS",
    "NAVER": "035420.KS"
}

@tool
def get_korean_stock_price(company_name_or_ticker: str) -> str:
    """
    특정 한국 주식 종목의 실시간 주가를 가져옵니다.
    입력값은 'SK하이닉스' 같은 기업명이나 '000660.KS' 같은 티커 코드 모두 가능합니다.
    """
    # 딕셔너리에 있으면 티커로 변환, 없으면 입력값 그대로 사용
    ticker = TICKER_MAP.get(company_name_or_ticker, company_name_or_ticker)

    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'].iloc[-1]
        return f"현재 주가는 {int(price)}원 입니다."
    except Exception as e:
        return f"주가 정보를 가져오는 데 실패했습니다. (AI가 입력한 값: {company_name_or_ticker})"

@tool
def get_stock_history(company_name_or_ticker: str) -> str:
    """
    특정 주식의 최근 1달간의 주가 흐름(최고가, 최저가, 시작가, 현재가)을 가져옵니다.
    입력값은 'SK하이닉스' 같은 기업명이나 '000660.KS' 같은 티커 코드 모두 가능합니다.
    """
    # 여기에도 동일한 변환 로직 적용
    ticker = TICKER_MAP.get(company_name_or_ticker, company_name_or_ticker)
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        max_price = int(hist['High'].max())
        min_price = int(hist['Low'].min())
        start_price = int(hist['Close'].iloc[0])
        end_price = int(hist['Close'].iloc[-1])
        trend = "상승세" if end_price > start_price else "하락세"
        return f"최근 1달: 시작가 {start_price}원 -> 현재가 {end_price}원 ({trend}). 최고가 {max_price}원, 최저가 {min_price}원."
    except Exception as e:
        return f"과거 주가 데이터를 가져오지 못했습니다. (AI가 입력한 값: {company_name_or_ticker})"
@tool
def search_news(query: str) -> str:
    """특정 기업의 최신 뉴스를 웹에서 검색합니다. '기업명 최신 뉴스' 형태로 검색하세요."""
    try:
        # 💡 [수정 3] Tavily 검색기를 사용하도록 변경 (가장 관련성 높은 결과 3개만 가져옴)
        search = TavilySearchResults(max_results=3) 
        result = search.invoke(query)
        if not result:
            return "뉴스를 찾을 수 없습니다."
        # Tavily는 JSON 형태의 리스트를 반환하므로 문자열로 변환해서 줍니다.
        return str(result) 
    except Exception as e:
        return f"뉴스 검색 시스템 오류 원인: {str(e)}"

tools = [calculate_average, multiply, get_today_date, get_korean_stock_price, get_stock_history, search_news]


# ==========================================
# 💡 핵심: 대화 기록을 저장하는 메모리
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 💬 채팅 입력창 및 에이전트 실행부
# ==========================================
if prompt := st.chat_input("예: SK하이닉스 최근 주가 분석해 줘"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent_memory = [(msg["role"], msg["content"]) for msg in st.session_state.messages]
    
    with st.chat_message("assistant"):
        with st.spinner("AI가 분석 중입니다..."):
            
            # 💡 수정 2: 실제로 존재하는 안정적인 모델명으로 변경
            models_to_try = [
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite"
            ]
            
            success = False
            
            for model_name in models_to_try:
                try:
                    current_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
                    agent_executor = create_react_agent(current_llm, tools)
                    
                    response = agent_executor.invoke({"messages": agent_memory})
                    
                    final_answer = response["messages"][-1].content
                    if isinstance(final_answer, list) and len(final_answer) > 0 and 'text' in final_answer[0]:
                        answer_text = final_answer[0]['text']
                    else:
                        answer_text = final_answer
                    
                    st.markdown(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                    
                    success = True
                    break # 성공했으니 반복문 탈출!
                    
                except Exception as e:
                    # 💡 [수정됨] 토스트 메시지 대신, 화면에 노란색 경고창으로 진짜 에러 내용을 크게 띄웁니다!
                    st.warning(f"🚨 {model_name} 실패 원인: {e}")
                    continue
            
            if not success:
                st.error("AI 에이전트 실행에 실패했습니다. 잠시 후 다시 시도해 주세요.")
