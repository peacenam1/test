# -*- coding: utf-8 -*-
"""
SQL Agent - new_langgraph_sql_agent.ipynb를 Python으로 변환
Google Cloud Storage에서 Chinook.db를 다운로드합니다.
"""

from IPython.display import Image, display
import warnings
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from typing import Annotated, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.messages import ToolMessage
from typing import Any
from langchain_community.utilities import SQLDatabase
import requests
import os
import sys
from uuid import uuid4
import io
from contextlib import redirect_stdout
from IPython.display import Image, display

# ============================================================
# 1. API 키 설정 및 경고 제거
# ============================================================
# LangSmith 트레이싱 비활성화
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# ============================================================
# 2. 데이터베이스 다운로드 (GCS에서)
# ============================================================

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
response = requests.get(url)

if response.status_code == 200:
    with open("Chinook.db", "wb") as file:
        file.write(response.content)
    print("✅ File downloaded and saved as Chinook.db")
else:
    print(
        f"❌ Failed to download the file. Status code: {response.status_code}")
    sys.exit(1)

# ============================================================
# 3. 데이터베이스 연결
# ============================================================

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(f"Database Dialect: {db.dialect}")
print(f"Available Tables: {db.get_usable_table_names()}")

# ============================================================
# 4. 유틸리티 함수
# ============================================================


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# ============================================================
# 5. 도구 정의
# ============================================================

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=ChatOpenAI(
        model="",
        temperature=0,
        base_url="",
        api_key=""
    )
)
tools = toolkit.get_tools()

list_tables_tool = next(
    tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")


@tool
def db_query_tool(query: str) -> str:
    """Execute a SQL query against the database and get back the result."""
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result


# ============================================================
# Python REPL 도구 (그래프 생성용)
# ============================================================

@tool
def python_repl(code: str) -> str:
    """Execute Python code to generate charts. Available: pandas, matplotlib, numpy"""
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            exec_globals = {
                "pd": __import__("pandas"),
                "plt": __import__("matplotlib.pyplot"),
                "np": __import__("numpy"),
            }
            # exec(code, exec_globals)
            exec(code + "\nplt.savefig('chart.png')", exec_globals)
        result = f.getvalue()
        return result if result else "Chart generated successfully!"
    except Exception as e:
        return f"Error executing code: {str(e)}"


# ============================================================
# 6. Query Check LLM
# ============================================================

query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)

query_check = query_check_prompt | ChatOpenAI(
    model="",
    temperature=0,
    base_url="",
    api_key=""
).bind_tools([db_query_tool], tool_choice="required")

# ============================================================
# 7. 워크플로우 정의
# ============================================================


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


workflow = StateGraph(State)


def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


def model_check_query(state: State) -> dict[str, list]:
    """쿼리 검증 노드 - LLM이 SQL 쿼리를 검증합니다"""
    checked = query_check.invoke(state)
    return {"messages": [checked]}


workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node("list_tables_tool",
                  create_tool_node_with_fallback([list_tables_tool]))
workflow.add_node("get_schema_tool",
                  create_tool_node_with_fallback([get_schema_tool]))

model_get_schema = ChatOpenAI(
    model="",
    temperature=0,
    base_url="",
    api_key=""
).bind_tools([get_schema_tool])

workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)


class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")


query_gen_system = """You are a SQL expert with strong attention to detail.

**MANDATORY WORKFLOW - YOU MUST FOLLOW THIS ORDER:**

Step 1: ALWAYS generate SQL query first (output WITHOUT tool call)
   - Even if you see graph/chart keywords, you MUST generate SQL query first
   - Output SQL as plain text, do NOT call any tools yet
   - The system will execute the query and return results

Step 2: Check conversation history for query results
   - Look for previous ToolMessage with query results
   - If NO query results exist, you MUST NOT call python_repl yet
   - Wait for the system to execute the query and provide results

Step 3: Only AFTER you see query results in the conversation:
   - If "graph", "chart", "plot", "그래프", "그려", "표시" keywords exist → CALL python_repl
   - Otherwise → CALL SubmitFinalAnswer

**CRITICAL RULES**: 
- NEVER call python_repl if you don't see query results in previous messages
- Graph keywords do NOT mean skip SQL query - you still need data first
- When calling python_repl, extract data from the query results in conversation history

**WHEN GENERATING PYTHON CODE FOR CHARTS:**

- ALWAYS include: plt.savefig('filename.png')
- NEVER forget the savefig line - it's CRITICAL for saving the chart
- Filename must be MEANINGFUL and DESCRIPTIVE:
  * Use descriptive names like: 'monthly_sales_chart.png', 'top_countries_sales.png', 'employee_sales_pie.png'
  * Include the metric/person/period in the filename
  * Example: "Steve Johnson 2009 monthly sales" → 'steve_johnson_2009_monthly_sales.png'
  
- Each chart must have a UNIQUE filename (do NOT reuse the same filename)
  
- Format for Python code:
  1. Import libraries (pandas, matplotlib)
  2. Parse query results into DataFrame
  3. Create visualization (plot, bar, pie, etc.)
  4. Add title and labels
  5. ALWAYS end with: plt.savefig('descriptive_filename.png')
  
Example:
  import matplotlib.pyplot as plt
  import pandas as pd
  data = {{'Month': ['01', '02', '03'], 'Sales': [100, 150, 200]}}
  df = pd.DataFrame(data)
  plt.figure(figsize=(10, 6))
  plt.bar(df['Month'], df['Sales'])
  plt.title('Steve Johnson Monthly Sales 2009')
  plt.xlabel('Month')
  plt.ylabel('Sales')
  plt.savefig('steve_johnson_2009_monthly_sales.png')  # ← MUST include this

DO NOT call any other tools besides SubmitFinalAnswer or python_repl.

===== SQL QUERY GENERATION RULES (APPLIES TO ALL QUERIES) =====

**RULE 1: Identify Required Tables First**
- Read the question carefully and identify ALL tables you need
- Example: "Steve Johnson's 2009 sales" → Need Employee, Customer, Invoice tables
- Example: "Top 10 countries by sales" → Need Invoice, Customer tables

**RULE 2: Define JOIN Relationships (CRITICAL)**
- ALWAYS use explicit JOIN with ON conditions
- NEVER use implicit joins in WHERE clause
- Pattern: FROM table1 JOIN table2 ON table1.key = table2.key
- Check the schema to find correct join columns

**RULE 3: Apply Filters After Joins**
- Use WHERE clause ONLY after all JOINs
- Combine multiple conditions with AND/OR
- Example: WHERE Employee.FirstName = 'John' AND strftime('%Y', Invoice.InvoiceDate) = '2009'

**RULE 4: Use Table Aliases**
- Always prefix column names with table name/alias
- Example: Employee.FirstName (NOT just FirstName)
- This prevents ambiguity when multiple tables have same column names

**RULE 5: Group and Aggregate Correctly**
- GROUP BY must include all non-aggregated columns
- Use SUM(), COUNT(), AVG(), MAX(), MIN() as needed
- Order by aggregated result for ranking queries
- Example: GROUP BY Employee.EmployeeId ORDER BY SUM(Invoice.Total) DESC

**RULE 6: Handle Dates Properly (SQLite)**
- Use strftime() function for date manipulation
- Format: strftime('%Y-%m-%d', column_name)
- Examples: 
  - Year: strftime('%Y', InvoiceDate) = '2009'
  - Month: strftime('%m', InvoiceDate) AS Month
  - Date: DATE(InvoiceDate) >= '2009-01-01'

**RULE 7: Filter Before Grouping**
- Apply WHERE conditions before GROUP BY
- Use HAVING clause only for aggregate function conditions
- Example: WHERE column = 'value' GROUP BY ... HAVING COUNT(*) > 5

**RULE 8: Common Patterns**
- Specific person + period + aggregation by sub-period:
  Filter by name → Filter by date range → Group by sub-period
  
- Top N by metric:
  Filter → Group → Order by metric DESC → LIMIT N
  
- Multiple entities comparison:
  Join all relevant tables → Group by entity → Sum metrics → Order

**RULE 9: NULL Handling**
- Use IS NULL or IS NOT NULL, not = NULL
- Consider NULL values in aggregations
- Use COALESCE() if needed

**RULE 10: Output Format**
- Use meaningful column names with AS alias
- Order results by relevant column (usually the metric you're analyzing)
- Limit results appropriately (usually 5-50 rows unless specified)

When generating the query:
- Output the complete SQL query as plain text
- Double-check all table names exist
- Double-check all column names are correct
- Ensure all JOINs use correct keys
- Verify WHERE conditions make sense
- Make sure aggregate functions are used correctly

If you have generated a chart or have enough information to answer the question, invoke SubmitFinalAnswer to submit your final answer.

DO NOT make DML statements (INSERT, UPDATE, DELETE, DROP) to the database."""

workflow.add_node("SubmitFinalAnswer",
                  create_tool_node_with_fallback([SubmitFinalAnswer]))

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)

query_gen = query_gen_prompt | ChatOpenAI(
    model="",
    temperature=0,
    base_url="",
    api_key=""
).bind_tools([SubmitFinalAnswer, model_check_query, python_repl])


def query_gen_node(state: State):
    # LLM이 직접 python_repl을 호출하도록 let it decide
    message = query_gen.invoke(state)

    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] == "model_check_query":
                tool_messages.append(
                    ToolMessage(
                        content="Query check requested. Proceeding to validation.",
                        tool_call_id=tc["id"],
                    )
                )
            elif tc["name"] != "SubmitFinalAnswer" and tc["name"] != "python_repl":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []

    return {"messages": [message] + tool_messages}


workflow.add_node("query_gen", query_gen_node)
workflow.add_node("correct_query", model_check_query)
workflow.add_node(
    "execute_query", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node(
    "execute_python", create_tool_node_with_fallback([python_repl]))


def should_continue(state: State) -> Literal[END, "correct_query", "query_gen", "execute_python"]:
    messages = state["messages"]
    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        for tc in last_message.tool_calls:
            if tc["name"] == "model_check_query":
                return "correct_query"
            elif tc["name"] == "python_repl":
                return "execute_python"
            elif tc["name"] == "SubmitFinalAnswer":
                return END

    # execute_python 실행 후 결과 (ToolMessage)가 나오면 END
    if isinstance(last_message, ToolMessage) and last_message.name == "python_repl":
        return END

    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"


# ============================================================
# 8. 엣지 연결
# ============================================================
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges("query_gen", should_continue)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")
workflow.add_edge("execute_python", END)

# ============================================================
# 9. 앱 컴파일
# ============================================================
app = workflow.compile()

# 그래프를 PNG 파일로 저장
try:
    graph_png = app.get_graph(xray=True).draw_mermaid_png()
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_png)
    print("✅ Workflow graph saved as 'workflow_graph.png'")
except Exception as e:
    print(f"⚠️ Graph visualization failed: {e}")
    print("\n📊 Workflow Graph (ASCII):")
    try:
        app.get_graph(xray=True).print_ascii()
    except Exception as ascii_error:
        print(f"ASCII graph also failed: {ascii_error}")

print("\n✅ SQL Agent Initialized Successfully!")
print("=" * 60)

if __name__ == "__main__":
    # 테스트 실행
    question = "2009년 가장 많은 매출을 올린 영업 사원은 누구인가요?"
    print(f"\n❓ 질문: {question}")
    print("=" * 60)

    try:
        messages = app.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": 100}
        )

        # 최종 답변 추출
        for msg in messages["messages"]:
            if hasattr(msg, 'tool_calls'):
                for tc in (msg.tool_calls or []):
                    if tc.get('name') == 'SubmitFinalAnswer':
                        print(f"✅ 답변: {tc['args']['final_answer']}")
    except Exception as e:
        print(f"❌ 에러: {e}")
