# Agentic RAG

> **本项目基于 [luisgdev/agentic-rag](https://github.com/luisgdev/agentic-rag) 二次开发。**
> 原项目采用 Mistral + CPU Embedding，本仓库在其基础上完成了 LLM 替换、Embedding GPU 加速、中文切分优化等改造。
> 原始代码版权归原作者 Luis Ch. 所有，遵循 MIT 许可证（见 [LICENSE](LICENSE)）。

一个基于多 Agent 架构的检索增强生成（RAG）问答系统，使用 FastAPI + LangChain + DeepSeek + ChromaDB 构建。

## 项目亮点

- **多 Agent 架构**：检索 Agent（Retriever）与响应 Agent（Responder）分工协作，由响应 Agent 自主决定是否触发检索
- **DeepSeek 大模型接入**：通过 LangChain 的 OpenAI 兼容接口调用 DeepSeek API
- **GPU 加速的 Embedding**：将 `bge-small-zh-v1.5` 从 CPU 迁移到 NVIDIA RTX 4060，查询延迟从 1.9ms 降至 0.5ms
- **中文文本切分优化**：针对中文缺少空格的特点，自定义切分分隔符，保证语义完整性
- **异步处理 + Webhook 回调**：FastAPI + asyncio.Queue 实现请求入队与后台处理，RAG 结果通过 Webhook 异步推送
- **向量检索**：基于 ChromaDB 的语义检索，支持中英文混合查询

## 架构

系统由两个核心 Agent 组成：

```
用户查询
   ↓
[Responder Agent] ── 判断是否需要检索
   ↓ 需要
[Retriever Agent] ── 语义检索
   ↓
[ChromaDB] ← [bge-small-zh-v1.5 on GPU]
   ↓
[DeepSeek] ── 生成最终回答
   ↓
Webhook 回调
```

- **Retriever Agent**：负责在向量数据库中做语义检索，返回相关文档块
- **Responder Agent**：负责与用户交互，自主决策是否调用检索工具，最终生成回答

## 技术栈

| 组件 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| Agent 编排 | LangChain |
| 大语言模型 | DeepSeek（`deepseek-chat`） |
| Embedding | BAAI/bge-small-zh-v1.5（GPU 加速） |
| 向量数据库 | ChromaDB |
| 配置管理 | pydantic-settings |
| 依赖管理 | uv |
| 运行环境 | Python 3.12 + CUDA 13.2 + PyTorch 2.12 |

## 快速开始

### 前置要求

- Python 3.12+
- NVIDIA GPU（可选，CPU 也能运行，但速度较慢）
- DeepSeek API Key

### 安装

1. **克隆仓库**

   ```bash
   git clone https://github.com/Dhr5466/agentic-rag.git
   cd agentic-rag
   ```

2. **安装依赖（使用 uv）**

   ```bash
   uv sync
   ```

   如果使用 GPU，需要额外安装 CUDA 版 PyTorch：

   ```bash
   pip uninstall torch torchvision -y
   pip install torch==2.12.0 torchvision --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu132
   ```

3. **配置环境变量**

   ```bash
   cp .env.sample .env
   ```

   编辑 `.env`，填入你的 DeepSeek API Key：

   ```env
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

4. **下载 Embedding 模型**

   国内推荐使用 ModelScope：

   ```bash
   modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir ./models/bge-small-zh-v1.5
   ```

   然后在 `rag/vector.py` 中把 `model_name` 指向本地路径：

   ```python
   model_name=r"./models/bge-small-zh-v1.5",
   ```

5. **启动服务**

   ```bash
   uvicorn app:app --port 8000
   ```

   服务运行在 `http://localhost:8000`，交互式 API 文档见 `http://localhost:8000/docs`。

## API 使用

### POST `/query`

提交查询：

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "1427b3b0 这个产品的价格和库存是多少？",
       "user_id": "user123"
     }'
```

响应（异步，立即返回）：

```json
{ "message": "用户查询已成功加入队列！" }
```

RAG 生成的结果会通过 Webhook 回调推送到 `CALLBACK_URL`。

### POST `/webhook`

接收回调数据的端点（内部使用）。

## 配置项

`.env` 文件中的配置：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（必填） | — |
| `DOCUMENTS_DIRECTORY` | 知识库文档目录 | `documents` |
| `VECTOR_DB_DIRECTORY` | 向量库存储目录 | `.vector_db` |
| `CHUNK_SIZE` | 文档切分块大小 | `1000` |
| `CHUNK_OVERLAP_SIZE` | 块重叠大小 | `100` |
| `TOP_K` | 检索返回文档数 | `5` |
| `CALLBACK_URL` | Webhook 回调地址 | — |

## 项目结构

```
agentic-rag/
├── agents/                  # 多 Agent 模块
│   ├── __init__.py          # 大模型实例（DeepSeek）
│   ├── responder.py         # 响应 Agent：决策 + 生成回答
│   └── retriever.py         # 检索 Agent：向量检索
├── app/                     # FastAPI 应用层
│   ├── __init__.py          # 路由 + 队列 + 后台任务
│   └── utils.py             # 文档索引工具
├── documents/               # 知识库文档（.txt / .md）
├── rag/                     # RAG 核心
│   ├── __init__.py          # 门面：暴露 vector_store
│   └── vector.py            # Embedding + ChromaDB
├── tests/                   # 单元测试
├── config.py                # 配置管理
├── pyproject.toml           # 项目依赖
└── uv.lock                  # 依赖锁定
```

## 性能数据

在 RTX 4060 Laptop GPU + i9 上的 Embedding 测试结果（512 条文本）：

| 模型 | 设备 | 单条延迟 | 批量吞吐（batch=32） |
|---|---|---|---|
| bge-small-zh-v1.5 | CPU | 1.9 ms | 1091 条/s |
| bge-small-zh-v1.5 | **GPU** | **0.5 ms** | **3562 条/s** |
| bge-m3 | CPU | 17.1 ms | 42.7 条/s |
| bge-m3 | GPU | 5.2 ms | 268.7 条/s |

本项目选择 `bge-small-zh-v1.5` + GPU 组合：模型轻量、中文效果优秀、GPU 加速后单条延迟 0.5ms，满足 RAG 实时查询需求。

## 与原项目的区别

本项目在原 [agentic-rag](https://github.com/luisgdev/agentic-rag) 基础上做了以下改造：

- **LLM 替换**：Mistral → DeepSeek，通过 `langchain-openai` 的 `ChatOpenAI` 指向 DeepSeek API
- **Embedding 加速**：`bge-small-zh-v1.5` 从 CPU 迁移到 GPU，查询延迟降低约 4 倍
- **中文切分优化**：自定义 `separators`，优先按中文标点（。！？；，）切分
- **依赖管理**：使用 `uv` + 南大 CUDA 镜像源，实现 GPU 版 PyTorch 的可复现安装
- **环境隔离**：每个项目独立 `.venv`，避免依赖冲突

## License

详见 [LICENSE](LICENSE) 文件。
