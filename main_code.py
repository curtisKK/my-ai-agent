import os
import datetime
import yfinance as yf
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. 웹사이트 UI 설정
st.title("📈 AI 주식 분석 에이전트")
st.markdown("질문을 입력하면 AI가 실시간 주가를 검색하고 계산해 줍니다.")

# 🚨 2. 보안 핵심: API 키를 코드에 적지 않고 Streamlit 금고(Secrets)에서 불러옵니다.
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

tools = [get_korean_stock_price, calculate_average, multiply, get_today_date]

# 4. 에이전트 설정 (무료 한도가 넉넉한 1.5-flash 사용 권장)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
agent_executor = create_react_agent(llm, tools)

# 5. 사용자 입력 및 실행 버튼 UI
query = st.text_input("질문을 입력하세요:", "오늘 날짜와 현재 삼성전자(005930.KS)와 SK하이닉스(000660.KS)의 실시간 주가를 찾아봐줘.")

if st.button("에이전트 실행"):
    with st.spinner("AI가 실시간 데이터를 검색하고 계산 중입니다. 잠시만 기다려주세요..."):
        try:
            response = agent_executor.invoke({"messages": [("user", query)]})
            final_answer = response["messages"][-1].content
            
            st.success("답변 생성 완료!")
            st.markdown("### 🤖 에이전트 최종 답변")
            if isinstance(final_answer, list) and len(final_answer) > 0 and 'text' in final_answer[0]:
                st.info(final_answer[0]['text'])
            else:
                st.info(final_answer)
        except Exception as e:
            st.error(f"에러가 발생했습니다: {e}")
