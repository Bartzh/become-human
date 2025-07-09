from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
from fastapi.security import OAuth2PasswordBearer
import jwt
import asyncio, uvicorn
import bcrypt
from dotenv import load_dotenv
import os
import json
from typing import Annotated, Optional

from langchain_core.tools import tool
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.runnables import RunnableConfig
#from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_qwq import ChatQwen

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import AnyMessage, ToolMessage, BaseMessage, AIMessage, HumanMessage, RemoveMessage

from become_human.graph_main import MainGraph
from become_human.graph_recycle import RecycleGraph
from become_human.graph_retrieve import RetrieveGraph
from become_human.memory import MemoryManager
from become_human.utils import extract_text_parts
from main import command_processing

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

import json, re

from datetime import datetime, timedelta, timezone

#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],  # 允许所有来源，根据需要调整为具体域名
#    allow_credentials=True,
#    allow_methods=["*"],  # 允许所有 HTTP 方法（包括 OPTIONS）
#    allow_headers=["*"],  # 允许所有请求头
#)


#config = {"configurable": {"thread_id": "1"}}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# 用户数据文件路径
USERS_FILE = "./config/app_users.json"
DEFAULT_USERS = {
    "default_user": {
        "password": "donotchangeifyouwannaletmein",
        "is_admin": True,
        "accessible_threads": ["default_thread"]
    }
}


def load_users_from_json() -> dict:
    """从 users.json 文件中加载用户信息，若文件不存在则创建空文件"""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump(DEFAULT_USERS, f, indent=4)
            return {}

    with open(USERS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


users_db = load_users_from_json()


load_dotenv()
private_key = os.getenv("APP_PRIVATE_KEY", "become-human")


@tool(response_format="content_and_artifact")
async def retrieve_memories(search_string: Annotated[str, "要检索的内容"], config: RunnableConfig) -> str:
    """从数据库（大脑）中检索记忆"""
    result = await retrieve_graph.graph.ainvoke({"input": search_string, "type": "active"}, config)
    content = result["output"]
    artifact = {"dont_store": True}
    return content, artifact

async def init():
    global llm, embeddings, main_graph, recycle_graph, retrieve_graph, memory_manager

    envs = ["CHAT_MODEL_NAME", "CHAT_MODEL_ENABLE_THINKING", "STRUCTURED_MODEL_NAME", "STRUCTURED_MODEL_ENABLE_THINKING"]
    for e in envs:
        if not os.getenv(e):
            raise Exception(f"{e} is not set")

    llm = ChatQwen(
        #model="qwen-max-2025-01-25",
        model=os.getenv("CHAT_MODEL_NAME"),
        #top_p=0.8,
        max_retries=2,
        enable_thinking=True if os.getenv("CHAT_MODEL_ENABLE_THINKING") == "true" else False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #parallel_tool_calls=True
    )

    llm_no_thinking = ChatQwen(
        #model="qwen-max-2025-01-25",
        model=os.getenv("STRUCTURED_MODEL_NAME"),
        #top_p=0.8,
        max_retries=2,
        enable_thinking=True if os.getenv("STRUCTURED_MODEL_ENABLE_THINKING") == "true" else False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    embeddings = DashScopeEmbeddings(model="text-embedding-v3")

    memory_manager = await MemoryManager.create(embeddings)

    retrieve_graph = await RetrieveGraph.create(llm_no_thinking, memory_manager)
    main_graph = await MainGraph.create(llm, retrieve_graph, memory_manager, [retrieve_memories], llm_no_thinking)
    recycle_graph = await RecycleGraph.create(llm_no_thinking, memory_manager)


@app.get("/api/get_accessible_threads")
async def get_accessible_threads(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    user_id = payload["sub"]
    print(users_db[user_id]['accessible_threads'])
    return {'accessible_threads': users_db[user_id]['accessible_threads']}


@app.post("/api/init")
async def init_endpoint(request: Request, token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    api_input = await request.json()
    thread_id = api_input.get("thread_id")
    user_id = payload['sub']
    await verify_thread_accessible(user_id, thread_id)
    #print(thread_id)
    await memory_manager.init_thread(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    main_state = await main_graph.graph.aget_state(config)
    main_messages: list[AnyMessage] = main_state.values.get("messages", [])
    human_message_pattern = re.compile(r'^\[.*?\]\n.*?: ')
    messages = []
    for message in main_messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call["name"] == "send_message":
                    messages.append({"role": "ai", "content": tool_call["args"]["message"], "id": f'{message.id}.{tool_call["id"]}', "name": None})
        elif isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                content = human_message_pattern.sub('', message.text())
                messages.append({"role": message.type, "content": content, "id": message.id, "name": message.name})
            elif isinstance(message.content, list):
                count = 0
                for c in message.content:
                    if isinstance(c, str):
                        content = human_message_pattern.sub('', c)
                        messages.append({"role": message.type, "content": content, "id": f'{message.id}.{count}', "name": message.name})
                    elif isinstance(c, dict):
                        if c.get("type") == "text" and isinstance(c.get("text"), str):
                            content = human_message_pattern.sub('', c["text"])
                            messages.append({"role": message.type, "content": content, "id": f'{message.id}.{count}', "name": message.name})
                    count += 1
        else:
            messages.append({"role": message.type, "content": message.text(), "id": message.id, "name": message.name})
    return {"messages": messages}

@app.post("/api/stream")
async def stream_endpoint(request: Request, token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    user_input: dict = await request.json()
    message = user_input.get("message")
    extracted_message = extract_text_parts(message)
    if not extracted_message:
        raise HTTPException(status_code=400, detail="message is required")
    user_id = payload['sub']
    thread_id = user_input.get("thread_id")
    await verify_thread_accessible(user_id, thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    is_admin = users_db[user_id].get('is_admin')
    if extracted_message[0].startswith("@"):
        if is_admin:
            log = await command_processing(thread_id, extracted_message[0], main_graph, recycle_graph, retrieve_graph)
            return Response(json.dumps(log, ensure_ascii=False) + '\n', media_type="application/json")
        else:
            return Response(json.dumps({"name": "log", "args": {"message": "无权限执行此命令"}}, ensure_ascii=False) + '\n', media_type="application/json")

    async def generate():
        async for item in main_graph.stream_graph_updates(extracted_message, config, user_name=user_input.get("user_name")):
            yield json.dumps(item, ensure_ascii=False) + '\n'
            #yield json.dumps(item)

        # 处理回收逻辑
        main_state = await main_graph.graph.aget_state(config)
        main_messages = main_state.values["messages"]
        new_messages = main_state.values["new_messages"]
        print(new_messages)
        print(f'{count_tokens_approximately(main_messages)} tokens')
        recycle_response = await recycle_graph.graph.ainvoke({"input_messages": main_messages}, config)
        if recycle_response.get("success"):
            await main_graph.graph.aupdate_state(config, {"messages": recycle_response["remove_messages"]})

    return StreamingResponse(generate(), media_type="application/json")



# 生成 JWT 的函数
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
        to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, private_key, algorithm="HS256")
    return encoded_jwt

# 验证 JWT 的函数
def verify_token(token: str):
    try:
        payload = jwt.decode(token, private_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    if payload['sub'] not in users_db.keys():
        raise HTTPException(status_code=400, detail="User not found")
    return payload

async def verify_thread_accessible(user_id: Optional[str] = None, thread_id: Optional[str] = None):
    if not user_id:
        raise HTTPException(status_code=400, detail="User id is required")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread id is required")
    if user_id not in users_db.keys():
        raise HTTPException(status_code=400, detail="User not found")
    if thread_id not in users_db[user_id]['accessible_threads']:
        raise HTTPException(status_code=400, detail="thread is not accessible")



@app.post("/api/login")
async def login(request: Request):
    r: dict = await request.json()
    username = r.get("username")
    u: dict = users_db.get(username)
    if not u:
        raise HTTPException(status_code=400, detail="User not found")
    pw: str = u.get("password")
    if not pw:
        raise HTTPException(status_code=400, detail="Password not found")
    hashedpassword: str = r.get("password")
    if not bcrypt.checkpw(pw.encode('utf-8'), hashedpassword.encode('utf-8')):
        print("Incorrect username or password")
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": username}, expires_delta=timedelta(weeks=2))
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/verify")
async def verify_route(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    return {"username": payload['sub']}



async def exit():
    await main_graph.conn.close()
    await recycle_graph.conn.close()
    await retrieve_graph.conn.close()
    print('app exited')

if __name__ == '__main__':
    asyncio.run(init())
    uvicorn.run(app, host='localhost', port=36262, workers=1)
    asyncio.run(exit())