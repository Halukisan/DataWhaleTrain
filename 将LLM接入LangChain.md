## 将LLM接入LangChain

我喜欢GLM大模型！！！

### ChatGLM模型调用参数

ChatGLM3-6B 的调用接口有两个，一个是chat接口，一个是stream_chat接口

**chat接口**的原型如下：

```python
def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
```



参数说明如下：

| 参数名           | 参数含义                                                     | 默认值             |
| ---------------- | ------------------------------------------------------------ | ------------------ |
| tokenizer        | 用于处理输入和输出文本的tokenizer对象。由前面的 AutoTokenizer.from_pretrained 调用返回的对象 | 由模型决定         |
| query            | str 类型，用户输入的任何文本                                 | 无                 |
| history          | List[Dict]，可选参数；对话历史，每一项都是一个字典，包含角色（'role'）和内容（'content'）。 | None               |
| role             | str, 可选参数；输入文本的角色，可以是'user'或者'assistant'。 | user               |
| max_length       | int, 可选；生成文本的最大长度。                              | 8192               |
| num_beams        | int, 可选；Beam搜索的宽度，如果值大于1，则使用Beam搜索       | 1                  |
| do_sample        | bool, 可选；是否从预测分布中进行采样，如果为True，则使用采样策略生成回复。 | True               |
| top_p            | float, 可选；用于控制生成回复的多样性                        | 0.8                |
| temperature      | float, 可选；控制生成文本的随机性的参数                      | 0.8                |
| logits_processor | LogitsProcessorList, 可选；用于处理和修改生成步骤中的logits的对象 | None               |
| **kwargs         | 其他传递给模型生成函数的参数                                 | 忽略或根据需要设置 |

返回值：

```
response (str): 模型的响应文本。
history (List[Dict]): 更新后的对话历史。
```

**stream_chat接口**的原型如下：

```python
def stream_chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
                    past_key_values=None,max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                    logits_processor=None, return_past_key_values=False, **kwargs):
```

参数说明如下：

| 参数名                 | 参数含义                                                     | 默认值             |
| ---------------------- | ------------------------------------------------------------ | ------------------ |
| tokenizer              | 用于处理输入和输出文本的tokenizer对象。由前面的 AutoTokenizer.from_pretrained 调用返回的对象 |                    |
| query                  | str，必须参数；用户输入的任何聊天文本。                      |                    |
| history                | List[Dict], 可选；对话历史，每一项都是一个字典，包含角色（'role'）和内容（'content'）。 | None               |
| role                   | str, 可选： 输入文本的角色，可以是'user'或者'assistant'。    | user               |
| past_key_values        | List[Tensor], 可选；用于transformer模型的过去的键值对        | None               |
| max_length             | int, 可选： 生成文本的最大长度.                              | 8192               |
| do_sample              | bool, 可选；是否从预测分布中进行采样                         | True               |
| top_p                  | float, 可选： 用于控制生成回复的多样性。                     | 0.8                |
| temperature            | float, 可选；控制生成文本的随机性的参数                      | 0.8                |
| logits_processor       | LogitsProcessorList, 可选；用于处理和修改生成步骤中的logits的对象。 | None               |
| return_past_key_values | bool, 可选): 是否返回过去的键值对，用于下一步的生成。        | False              |
| **kwargs               | 其他传递给模型生成函数的参数。                               | 忽略或根据需要设置 |

返回值：

```
response (str): 模型的响应文本。
history (List[Dict]): 更新后的对话历史。
past_key_values (List[Tensor], 可选): 如果return_past_key_values为True，返回用于下一步生成的过去的键值对。
```

### 关于提示模板Prompt

在开发大模型应用的时候，我们并不会直接将用户输入的问题直接扔给大模型去进行问答，而是将用户输入的问题添加到我们自己定义的模板中。

我们首先要定义一个自己的模板：

在代码的开头，先介绍一下**prompt的格式**,参考[ChatGLM3 对话(prompt)格式 - 简书 (jianshu.com)](https://www.jianshu.com/p/066631299d0e)

1. 其中的system是系统消息，设计上可以穿插于对话中，但目前规定只能出现在开头

2. human是用户，不会连续出现多个来自human的消息

3. assistant：AI 助手，在出现之前必须有一个来自human的信息。

4. observation：外部返回的结果，必须在assistant消息之后。

```
from langchain.prompts.chat import ChatPromptTemplate

template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
messages  = chat_prompt.format_messages(input_language="中文", output_language="英文", text=text)
messages
[SystemMessage(content='你是一个翻译助手，可以帮助我将 中文 翻译成 英文.'),
 HumanMessage(content='我带着比身体重的行李，游入尼罗河底，经过几道闪电 看到一堆光圈，不确定是不是这里。')]
```

接下来让我们调用定义好的`llm`和`messages`来输出回答：

```
output  = llm.invoke(messages)
output
AIMessage(content='I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.')
```

返回的结果是JSON格式的，如何转换为text格式呢？

### Output parser（输出解析器）

OutputParsers 可以

- 将 LLM 文本转换为结构化信息（例如 JSON）
- 将 ChatMessage 转换为字符串
- 将除消息之外的调用返回的额外信息（如 OpenAI 函数调用）转换为字符串

可以将刚刚的AIMessage传递给output_parser，这是一个BaseOutputParser类型的，可以接受字符串或者BaseMessage作为参数输入。

**StrOutputParser（）方法可以简单的将任何输入转换为字符串。**

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
output_parser.invoke(output)
```

### 使用LCEL语法的完整流程

LCEL（LangChain Expression Language，Langchain的表达式语言）它将不同的组件链接在一起，将一个组件的输出作为下一个组件的输入。

用法示例：

```python
chain = prompt | model | output_parser
```

上面代码中我们使用 LCEL 将不同的组件拼凑成一个链，在此链中，用户输入传递到提示模板，然后提示模板输出传递到模型，然后模型输出传递到输出解析器。

1. 获取输入变量
2. 传递给模板
3. 传递给模型
4. 解析器输出

```python
chain = chat_prompt | llm | output_parser
chain.invoke({"input_language":"中文", "output_language":"英文","text": text})
```

## 智谱清言

因为langChain不支持ChatGLM，我们需要自定义一个LLM

```python
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from zhipuai import ZhipuAI

# 继承自 langchain.llms.base.LLM
class ZhipuAILLM(LLM):
    # 默认选用 ERNIE-Bot-turbo 模型，即目前一般所说的百度文心大模型
    model: str = "glm-4"
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        
        def gen_glm_params(prompt):
            '''
            构造 GLM 模型请求参数 messages

            请求参数：
                prompt: 对应的用户提示词
            '''
            messages = [{"role": "user", "content": prompt}]
            return messages
        
        client = ZhipuAI(
            api_key=self.api_key
        )
     
        messages = gen_glm_params(prompt)
        response = client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature
        )

        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "generate answer error"


    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用Ennie API的默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            }
        # print(type(self.model_kwargs))
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Wenxin"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}
```

### 接入LangChain

```python
# 需要下载源码
from zhipuai_llm import ZhipuAILLM
from dotenv import find_dotenv, load_dotenv
import os

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息

zhipuai_model = ZhipuAILLM(model="chatglm_std", temperature=0, api_key=api_key)

zhipuai_model("你好，请你自我介绍一下！")
```

## 构建检索问答链

根据自己构建的向量数据库，对query查询的问题进行召回，并将召回结果和query结合起来构建prompt，输入到大模型中进行问答。

### 加载向量数据库

#### Chroma向量数据库

此处的embedding模型或者API要和你构建向量数据库时用的一样

```python
# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings

from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import os
import sys
sys.path.append("../C3 搭建知识库") # 将父目录放入系统路径中
_ = load_dotenv(find_dotenv())    # 从环境变量中加载你的 API_KEY
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
```

加载向量数据库，其中包含了 ../../data_base/knowledge_db 下多个文档的 Embedding

```python
# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 向量数据库持久化路径
persist_directory = '../C3 搭建知识库/data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")
向量库中存储的数量：20
```

#### Milvus向量数据库

我使用的是Milvus向量数据库，代码如下

安装Milvus的Python客户端库：

```
pip install pymilvus==2.0.0
```

版本号（2.0.0）是一个示例，应该使用与自己的Milvus服务版本兼容的客户端版本。

接下来，可以使用以下代码来加载Milvus向量数据库：

```python
from zhipu.ai.embeddings import ZhipuAIEmbeddings
from chroma import Chroma
import pymilvus as pm

# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# Milvus 服务地址和端口
milvus_host = 'localhost'
milvus_port = '19530'

# 连接到Milvus
client = pm.connect(host=milvus_host, port=milvus_port)

# 创建或连接到现有的Milvus向量数据库
# 这里的`collection_name`是您在Milvus中创建的集合的名称
collection_name = 'document_embeddings'

# 如果集合不存在，则创建集合
if not client.has_collection(collection_name):
    # 定义集合的参数，例如维度、索引类型等
    fields = [
        pm.FieldSchema(name="id", dtype=pm.DataType.INT64, is_primary=True, auto_id=True),
        pm.FieldSchema(name="embedding", dtype=pm.DataType.FLOAT_VECTOR, dim=128)
    ]
    # 创建集合
    status = client.create_collection(collection_name, fields)
    print(status)

# 准备向量数据和对应IDs
# 假设您已经有了文档的ID和对应的嵌入向量
documents = ...  # 您的文档ID列表
embeddings = ...  # 对应的嵌入向量列表

# 将向量数据插入到Milvus集合中
status = client.insert(collection_name, embeddings, documents)
print(status)

# 创建索引以优化搜索性能
status = client.create_index(collection_name, field_name="embedding", index_type="IVF_FLAT", params={"nlist": 1024})
print(status)

# 确保在结束时关闭Milvus客户端连接
client.disconnect()

# 向量数据库持久化路径
persist_directory = '../C3 搭建知识库/data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

# 从此点开始，可以使用vectordb进行查询和操作
```

### 相似性搜索

快速分词工具 tiktoken 包：`pip install tiktoken`

如下代码会在向量数据库中根据相似性进行检索，返回前 k 个最相似的文档

```python
question = "什么是prompt engineering?"
docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(docs)}")
```

```txt
检索到的内容数：3
```

打印一下检索到的内容

```python
for i, doc in enumerate(docs):
    print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n"
```

### 开始构建

首先调用API创建一个LLM

```
from langchain_openai import ChatOpenAI
import os 
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)

llm.invoke("请你自我介绍一下自己！")
```

**构建检索问答链**

```python
from langchain.prompts import PromptTemplate

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
```

再创建一个基于模板的检索链：

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
```

创建检索 QA 链的方法 RetrievalQA.from_chain_type() 有如下参数：

- llm：指定使用的 LLM
- 指定 chain type : RetrievalQA.from_chain_type(chain_type="map_reduce")，也可以利用load_qa_chain()方法指定chain type。
- 自定义 prompt ：通过在RetrievalQA.from_chain_type()方法中，指定chain_type_kwargs参数，而该参数：chain_type_kwargs = {"prompt": PROMPT}
- 返回源文档：通过RetrievalQA.from_chain_type()方法中指定：return_source_documents=True参数；也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）

### 测试效果

```python
question_1 = "什么是南瓜书？"
question_2 = "王阳明是谁？"
```

#### 大模型自己回答的效果

```python
prompt_template = """请回答下列问题:
                            {}""".format(question_1)

### 基于大模型的问答
llm.predict(prompt_template)
```

```tex
'南瓜书是指一种关于南瓜的书籍，通常是指介绍南瓜的种植、养护、烹饪等方面知识的书籍。南瓜书也可以指一种以南瓜为主题的文学作品。'
```

大模型对于新的知识回答的不是很好，我们可以通过构建本地知识库来解决，不过，最好是function calling来解决，缓解模型幻觉问题

## 模型记忆功能的实现

 LangChain 中的储存模块可以将先前的对话嵌入到语言模型中，使其具有连续对话的能力

使用 `ConversationBufferMemory` ，它保存聊天消息历史记录的列表，这些历史记录将在回答问题时与问题一起传递模型，从而将它们添加到上下文中。

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)
```

### 对话检索链

在检索问答链的基础上，增加了处理历史对话的能力。

1. 将之前的对话和新的问题合成为一个完整的问题（查询）
2. 在向量数据库中搜索该问题（查询）的相关文档
3. 获取结果，存储到记忆区
4. 用户在UI中查看对话流程

```python
from langchain.chains import ConversationalRetrievalChain

retriever=vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
question = "我可以学习到关于提示工程的知识吗？"
result = qa({"question": question})
print(result['answer'])
是的，您可以学习到关于提示工程的知识。本模块内容基于吴恩达老师的《Prompt Engineering for Developer》课程编写，旨在分享使用提示词开发大语言模型应用的最佳实践和技巧。课程将介绍设计高效提示的原则，包括编写清晰、具体的指令和给予模型充足思考时间等。通过学习这些内容，您可以更好地利用大语言模型的性能，构建出色的语言模型应用。
```

然后基于答案进行下一个问题“为什么这门课需要教这方面的知识？”：

```python
question = "为什么这门课需要教这方面的知识？"
result = qa({"question": question})
print(result['answer'])
这门课程需要教授关于Prompt Engineering的知识，主要是为了帮助开发者更好地使用大型语言模型（LLM）来完成各种任务。通过学习Prompt Engineering，开发者可以学会如何设计清晰明确的提示词，以指导语言模型生成符合预期的文本输出。这种技能对于开发基于大型语言模型的应用程序和解决方案非常重要，可以提高模型的效率和准确性。
```
