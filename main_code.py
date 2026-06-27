import os
import datetime
import yfinance as yf
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

# 1. 웹사이트 UI 설정 (화면을 조금 더 넓게 씁니다)
st.set_page_config(page_title="주식 AI 에이전트", page_icon="📈", layout="centered")
st.title("📈 대화형 주식 분석 AI 에이전트")
st.markdown("이전 대화를 기억합니다! 편하게 채팅하듯 질문해 보세요.")

# 🚨 2. API 키 보안 설정
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("API 키가 금고(Secrets)에 설정되지 않았습니다.")
    st.stop()

# 3. 도구(Tools) 정의
@tool
def get_korean_stock_price(ticker: str) -> str:
    """한국 주식 종목의 실시간 주가를 가져옵니다."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'].iloc[-1]
        return f"{ticker}의 현재 주가는 {int(price)}입니다."
    except Exception as e:
        return "주가 정보를 가져오는 데 실패했습니다."

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
    
    [중요 지시사항] 
    사용자가 '삼성전자'나 'SK하이닉스' 처럼 기업 이름만 부르더라도,
    당신(AI)의 지식을 활용해 해당 기업의 6자리 한국 주식 종목코드를 스스로 찾으세요.
    그 후 코스피 종목은 뒤에 '.KS', 코스닥은 '.KQ'를 붙여서 이 도구의 입력값(ticker)으로 사용해야 합니다.
    (예시: 사용자가 "카카오 주가 찾아줘" 라고 하면 -> '035720.KS' 를 입력하여 실행할 것)
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'].iloc[-1]
        return f"{ticker}의 현재 주가는 {int(price)}입니다."
    except Exception as e:
        return f"주가 정보를 가져오는 데 실패했습니다. (입력된 값: {ticker})"
@tool
def get_stock_history(ticker: str) -> str:
    """
    특정 주식의 최근 1달간의 주가 흐름(최고가, 최저가, 변동성 등) 데이터를 가져옵니다.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo") # 최근 1개월 데이터
        
        # 데이터 전처리: 너무 길면 AI가 헷갈리므로 핵심 요약만 전달
        max_price = int(hist['High'].max())
        min_price = int(hist['Low'].min())
        start_price = int(hist['Close'].iloc[0])
        end_price = int(hist['Close'].iloc[-1])
        trend = "상승세" if end_price > start_price else "하락세"
        
        return f"{ticker}의 최근 1달 흐름: 시작가 {start_price}원 -> 현재가 {end_price}원 ({trend}). 최고가는 {max_price}원, 최저가는 {min_price}원 이었습니다."
    except Exception as e:
        return "과거 주가 데이터를 가져오지 못했습니다."

# 뉴스 검색기 도구 생성
search_news = DuckDuckGoSearchRun(
    name="search_news",
    description="특정 기업의 최신 뉴스, 호재, 악재, 시장 동향 등을 웹에서 검색합니다.")

tools = [get_korean_stock_price, calculate_average, multiply, get_today_date, get_korean_stock_price, get_stock_history]

# 4. 에이전트 설정 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # 본인에게 작동했던 모델명으로 변경 가능
agent_executor = create_react_agent(llm, tools)

# ==========================================
# 💡 핵심: 대화 기록을 저장하는 메모리(session_state) 만들기
# ==========================================
if "messages" not in st.session_state:
    # 첫 접속 시 메모리(리스트)를 비워둔 상태로 생성합니다.
    st.session_state.messages = []

# 기존의 대화 기록을 화면에 순서대로 그려줍니다. (새로고침 되어도 유지됨)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 💬 채팅 입력창 (카카오톡처럼 맨 아래에 고정됨)
# ==========================================
if prompt := st.chat_input("예: 오늘 삼성전자 주가 찾아줘"):
    
    # 1. 사용자가 입력한 질문을 화면에 표시하고 메모리에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 에이전트에게 이전 대화 기록까지 싹 다 묶어서 전달 ("우리가 지금까지 이런 대화를 했어!")
    agent_memory = [(msg["role"], msg["content"]) for msg in st.session_state.messages]
    
    # 3. AI의 답변을 받아오고 화면에 표시
    with st.chat_message("assistant"):
        with st.spinner("AI가 실시간 데이터를 검색하고 계산 중입니다..."):
            try:
                # 단일 질문이 아닌, 전체 대화 기록(agent_memory)을 invoke에 넣습니다.
                response = agent_executor.invoke({"messages": agent_memory})
                
                final_answer = response["messages"][-1].content
                if isinstance(final_answer, list) and len(final_answer) > 0 and 'text' in final_answer[0]:
                    answer_text = final_answer[0]['text']
                else:
                    answer_text = final_answer
                
                st.markdown(answer_text)
                
                # 4. AI의 최종 답변도 메모리에 저장하여 다음 질문 때 기억하게 함
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
            except Exception as e:
                st.error(f"에러가 발생했습니다: {e}")
