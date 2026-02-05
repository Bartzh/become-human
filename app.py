import sys
import jwt
import asyncio, uvicorn
import bcrypt
import os
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Optional
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.security import OAuth2PasswordBearer

from pathlib import Path
import aiohttp
import aiosqlite
from webpush import WebPush, WebPushSubscription

from langchain_core.messages import AIMessage, HumanMessage

from become_human.plugins import AgentReminderPlugin, AgentTimeIncrementerPlugin
from become_human import agent_manager
from become_human.message import extract_text_parts, BHMessageMetadata
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT

#from fastapi.middleware.cors import CORSMiddleware

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stdout, level=log_level)
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="2 weeks",
    enqueue=True,
    level=log_level
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化数据库
    await init_db()
    await agent_manager.init_manager(plugins=[
        AgentReminderPlugin,
        AgentTimeIncrementerPlugin,
    ])
    event_listener_task = asyncio.create_task(event_listener(agent_manager.event_queue))
    yield
    event_listener_task.cancel()
    try:
        await event_listener_task
    except asyncio.CancelledError:
        pass
    await agent_manager.close_manager()

app = FastAPI(lifespan=lifespan)


# 1. 捕获所有未处理的 Exception（500 错误）
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.opt(exception=exc).error("未处理的服务器异常: {}", str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )

# 2. 捕获 HTTPException（如 404, 400 等）
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # 可选：只记录 5xx，或全部记录
    if exc.status_code >= 500:
        logger.error("HTTP 5xx 错误: {} {}", exc.status_code, exc.detail)
    else:
        logger.warning("客户端错误: {} {}", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# 3. 捕获请求验证错误（Pydantic）
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("请求参数校验失败: {}", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())}
    )


#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],  # 允许所有来源，根据需要调整为具体域名
#    allow_credentials=True,
#    allow_methods=["*"],  # 允许所有 HTTP 方法（包括 OPTIONS）
#    allow_headers=["*"],  # 允许所有请求头
#)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")


# 用户数据文件路径
USERS_FILE = "./config/app_users.json"
DEFAULT_USERS = {
    "default_user": {
        "password": "donotchangeifyouwantme",
        "is_admin": True,
        "accessible_agents": [
            "default_agent_1",
            "default_agent_2",
            "default_agent_3"
        ]
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


private_key = os.getenv("APP_PRIVATE_KEY", "become-human")


user_queues: dict[str, asyncio.Queue] = {}

async def event_listener(queue: asyncio.Queue):
    while True:
        event = await queue.get()
        for user_id in user_queues.keys():
            if event["agent_id"] in users_db[user_id]['accessible_agents']:
                await user_queues[user_id].put(event)


@app.get("/api/get_accessible_agents")
async def get_accessible_agents(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    user_id = payload["sub"]
    return {'accessible_agents': users_db[user_id]['accessible_agents']}


@app.post("/api/init")
async def init_endpoint(request: Request, token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    api_input = await request.json()
    agent_id = api_input.get("agent_id")
    user_id = payload['sub']
    await verify_agent_accessible(user_id, agent_id)
    user_queues[user_id] = asyncio.Queue()
    await agent_manager.init_agent(agent_id)
    main_messages = await agent_manager.main_graph.get_messages(agent_id)
    human_message_pattern = re.compile(r'^\[.*?\]\n.*?: ')
    messages = []
    for message in main_messages:
        bh_metadata = BHMessageMetadata.parse(message)
        if (
            bh_metadata.message_type != 'bh:user' and
            bh_metadata.message_type != 'bh:ai'
        ):
            continue
        elif isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call["name"] == SEND_MESSAGE:
                    if tool_call["args"].get(SEND_MESSAGE_CONTENT):
                        messages.append({"role": "ai", "content": tool_call["args"][SEND_MESSAGE_CONTENT], "id": f'{message.id}.{tool_call["id"]}', "name": None})
                    else:
                        logger.warning(f'{SEND_MESSAGE}意外的没有参数，可能是打断导致的概率问题，也可能就是单纯的大模型输出错误')
        elif isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                content = human_message_pattern.sub('', message.text)
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
            messages.append({"role": message.type, "content": message.text, "id": message.id, "name": message.name})
    return {"messages": messages}

@app.post("/api/input")
async def input_endpoint(request: Request, token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    user_input: dict = await request.json()
    message = user_input.get("message")
    extracted_message = extract_text_parts(message)
    if not extracted_message:
        raise HTTPException(status_code=400, detail="message is required")
    user_id = payload['sub']
    agent_id = user_input.get("agent_id")
    await verify_agent_accessible(user_id, agent_id)

    is_admin = users_db[user_id].get('is_admin')

    asyncio.create_task(agent_manager.call_agent_for_user_with_command(
        user_input=extracted_message,
        agent_id=agent_id,
        is_admin=is_admin,
        user_name=user_input.get("user_name")
    ))

    return Response()


@app.get("/api/sse")
async def sse(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    user_id = payload['sub']
    sse_heartbeat_string = "event: heartbeat\ndata: feelmyheartbeat\n\n"
    async def event_generator():
        connection_id = user_id + '_' + str(id(asyncio.current_task()))
        try:
            while True:
                # 从事件队列中获取消息，设置超时时间
                queue = user_queues.get(user_id)
                if queue:
                    try:
                        # 使用 asyncio.wait_for 设置超时，避免长时间阻塞
                        event = await asyncio.wait_for(queue.get(), timeout=2.5)
                        # 在自我调用时发送的消息会同时推送通知
                        if (
                            event.get("is_self_call") and
                            event.get("name") == "send_message" and
                            (user_sub := await get_user_subscriptions(user_id)) and
                            (message_content := event.get("args", {}).get("content", ""))
                        ):
                            message = wp.get(message=message_content, subscription=user_sub, ttl=600)
                            async with aiohttp.ClientSession() as session:
                                await session.post(
                                    url=str(user_sub.endpoint),
                                    data=message.encrypted,
                                    headers=message.headers,
                                )
                        yield f"event: message\ndata: {json.dumps(event)}\n\n"  # 按照 SSE 格式发送消息
                    except asyncio.TimeoutError:
                        # 发送心跳消息防止连接超时
                        yield sse_heartbeat_string
                else:
                    # 如果没有队列，也发送心跳防止连接超时
                    await asyncio.sleep(1)
                    yield sse_heartbeat_string

        except asyncio.CancelledError:
            logger.info(f"SSE连接被取消: {connection_id}")
            raise
        except Exception as e:
            logger.error(f"SSE连接异常: {connection_id}, 错误: {str(e)}")
            raise
        finally:
            logger.info(f"SSE连接已关闭: {connection_id}")

    # 添加更多防止缓存的头部
    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream", 
        headers={
            "X-Accel-Buffering": "no", 
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff"
        }
    )

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

async def verify_agent_accessible(user_id: Optional[str] = None, agent_id: Optional[str] = None):
    if not user_id:
        raise HTTPException(status_code=400, detail="User id is required")
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent id is required")
    if user_id not in users_db.keys():
        raise HTTPException(status_code=400, detail="User not found")
    if agent_id not in users_db[user_id]['accessible_agents']:
        raise HTTPException(status_code=400, detail="agent is not accessible")


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
        logger.info("Incorrect username or password")
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": username}, expires_delta=timedelta(weeks=2))
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/verify")
async def verify_route(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    return {"username": payload['sub']}



wp = WebPush(
    public_key=Path("./public_key.pem"),
    private_key=Path("./private_key.pem"),
    subscriber="admin@mail.com",
)


DATABASE_PATH = "./data/app_users.sqlite"

async def init_db():
    """初始化数据库和表"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                subscription TEXT
            )
        """)
        await db.commit()

async def get_user_subscriptions(user_id: str) -> Optional[WebPushSubscription]:
    """从数据库获取用户的订阅"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(
            "SELECT subscription FROM users WHERE user_id = ?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                subscription = WebPushSubscription.model_validate(row[0])
                return subscription
            return None

async def save_subscription(user_id: str, subscription: WebPushSubscription):
    """保存用户的订阅到数据库，新订阅会替换旧订阅"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO users
            (user_id, subscription)
            VALUES (?, ?)
            """,
            (
                user_id,
                subscription.model_dump_json()
            )
        )
        await db.commit()

@app.get("/api/notification/key")
async def get_public_key(token: str = Depends(oauth2_scheme)):
    verify_token(token)
    if os.path.exists("./applicationServerKey"):
        with open("./applicationServerKey", "r") as f:
            return {"key": f.read()}
    raise HTTPException(status_code=404, detail="applicationServerKey not found")


@app.post("/api/notification/subscribe")
async def subscribe_user(subscription: WebPushSubscription, token: str = Depends(oauth2_scheme)):
    # global subscriptions
    payload = verify_token(token)
    user_id = payload['sub']
    
    # 保存订阅到数据库
    await save_subscription(user_id, subscription)
    
    return Response()



if __name__ == '__main__':
    uvicorn.run(app, host=os.getenv('APP_HOST', 'localhost'), port=int(os.getenv('APP_PORT', 36262)), workers=1)
