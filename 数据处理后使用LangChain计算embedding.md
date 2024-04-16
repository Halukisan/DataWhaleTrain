## 数据处理后使用LangChain计算embedding

* 第一部分包含数据处理（读取）、数据清洗、数据分割。

* 第二部分介绍py代码，对处理后的数据进行embedding值计算（使用LangChain框架，使用GLM的embeddingAPI）

### 数据处理

 LangChain 的 PyMuPDFLoader 来读取知识库的 PDF 文件。PyMuPDFLoader 是 PDF 解析器中速度最快的一种，结果会包含 PDF 及其页面的详细元数据，并且每页返回一个文档。

```python
rom langchain.document_loaders.pdf import PyMuPDFLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()
```



文档加载后储存在 `pages` 变量中:

- `page` 的变量类型为 `List`
- 打印 `pages` 的长度可以看到 pdf 一共包含多少页

```python
print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")
```



```tex
载入后的变量类型为：<class 'list'>， 该 PDF 一共包含 196 页
```

我们可以以几乎完全一致的方式读入 markdown 文档：

```python
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("../../data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md")
md_pages = loader.load()
```



读取的对象和 PDF 文档读取出来是完全一致的：

```python
print(f"载入后的变量类型为：{type(md_pages)}，",  f"该 Markdown 一共包含 {len(md_pages)} 页")
```



```tex
载入后的变量类型为：<class 'list'>， 该 Markdown 一共包含 1 页
```



```python
md_page = md_pages[0]
print(f"每一个元素的类型：{type(md_page)}.", 
    f"该文档的描述性数据：{md_page.metadata}", 
    f"查看该文档的内容:\n{md_page.page_content[0:][:200]}", 
    sep="\n------\n")
```

`page` 中的每一元素为一个文档，变量类型为 `langchain_core.documents.base.Document`, 文档变量类型包含两个属性

- `page_content` 包含该文档的内容。
- `meta_data` 为文档相关的描述性数据。

```python
pdf_page = pdf_pages[1]
print(f"每一个元素的类型：{type(pdf_page)}.", 
    f"该文档的描述性数据：{pdf_page.metadata}", 
    f"查看该文档的内容:\n{pdf_page.page_content}", 
    sep="\n------\n")
```

### 数据清洗

正则表达式匹配并删除掉`\n`

```python
import re
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
print(pdf_page.page_content)
```

删除掉`•`和空格

```python
pdf_page.page_content = pdf_page.page_content.replace('•', '')
pdf_page.page_content = pdf_page.page_content.replace(' ', '')
print(pdf_page.page_content)
```

删除换行符

```python
md_page.page_content = md_page.page_content.replace('\n\n', '\n')
print(md_page.page_content)
```

**关于excel文件类型的数据，可以参考**[Halukisan/DataClean: 模型训练Excel数据的清理 (github.com)](https://github.com/Halukisan/DataClean)

### 文档分割

由于单个文档的长度往往会超过模型支持的上下文，导致检索得到的知识太长超出模型的处理能力，将单个文档按长度或者按固定的规则分割成若干个 chunk，然后将每个 chunk 转化为词向量，存储到向量数据库中。

以 chunk 作为检索的元单位，也就是每一次检索到 k 个 chunk 作为模型可以参考来回答用户问题的知识，这个 k 是我们可以自由设定的。

Langchain 中文本分割器都根据 `chunk_size` (块大小)和 `chunk_overlap` (块与块之间的重叠大小)进行分割。

```python
#导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter
```



```python
# 知识库中单段文本长度
CHUNK_SIZE = 500

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50
```



```python
# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
text_splitter.split_text(pdf_page.page_content[0:1000])
```

```python
split_docs = text_splitter.split_documents(pdf_pages)
print(f"切分后的文件数量：{len(split_docs)}")
```



```tex
切分后的文件数量：720
```



```python
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
```



```tex
切分后的字符数（可以用来大致评估 token 数）：308931
```







### embedding计算

要实现自定义的embedding，需要定义一个自定义类继承自LangChain的Embedding基类，然后定义三个函数

1. _embed方法，接受一个字符串，并返回一个存放Embeddings的List[float]，即模型的核心调用
2. embed_query方法，用于对单个字符串进行embedding
3. embed_documents方法，用于对字符串列表（documents）进行embedding

首先我们导入所需的第三方库：

```python
from __future__ import annotations

import logging
from typing import Dict, List, Any

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)
```

这里我们定义一个继承自 Embeddings 类的自定义 Embeddings 类：

```python
class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""

    client: Any
    """`zhipuai.ZhipuAI"""
```

root_validator 用于在校验整个数据模型之前对整个数据模型进行自定义校验，以确保所有的数据都符合所期望的数据结构。

root_validator 接收一个函数作为参数，该函数包含需要校验的逻辑。函数应该返回一个字典，其中包含经过校验的数据。如果校验失败，则抛出一个 ValueError 异常。

我们只需将`.env`文件中`ZHIPUAI_API_KEY`配置好即可，`zhipuai.ZhipuAI`会自动获取`ZHIPUAI_API_KEY`。

```python
@root_validator()
def validate_environment(cls, values: Dict) -> Dict:
    """
    实例化ZhipuAI为values["client"]

    Args:

        values (Dict): 包含配置信息的字典，必须包含 client 的字段.
    Returns:

        values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
    """
    from zhipuai import ZhipuAI
    values["client"] = ZhipuAI()
    return values
```



接下来我们重写 `_embed` 方法,调用远程 API 并解析 embedding 结果。

```python
def _embed(self, texts: str) -> List[float]:
    embeddings = self.client.embeddings.create(
        model="embedding-2",
        input=texts
    )
    return embeddings.data[0].embedding
```

上面的代码是调用的国外的embedding模型，我建议使用国内（使用下面的代码）的，GLM最强！

这里的API选用`text-embedding-3-small`有着较好的性能跟价格，当我们预算有限时可以选择该模型；

智谱有封装好的SDK，我们调用即可。

```python
from zhipuai import ZhipuAI
def zhipu_embedding(text: str):

    api_key = os.environ['ZHIPUAI_API_KEY']
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response

text = '要生成 embedding 的输入文本，字符串形式。'
response = zhipu_embedding(text=text)
```



response为`zhipuai.types.embeddings.EmbeddingsResponded`类型，我们可以调用`object`、`data`、`model`、`usage`来查看response的embedding类型、embedding、embedding model及使用情况。

```python
print(f'response类型为：{type(response)}')
print(f'embedding类型为：{response.object}')
print(f'生成embedding的model为：{response.model}')
print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
print(f'embedding（前10）为: {response.data[0].embedding[:10]}')
```



```tex
response类型为：<class 'zhipuai.types.embeddings.EmbeddingsResponded'>
embedding类型为：list
生成embedding的model为：embedding-2
生成的embedding长度为：1024
embedding（前10）为: [0.017892399802803993, 0.0644201710820198, -0.009342825971543789, 0.02707476168870926, 0.004067837726324797, -0.05597858875989914, -0.04223804175853729, -0.03003198653459549, -0.016357755288481712, 0.06777040660381317]
```

重写 embed_documents 方法，因为这里 `_embed` 已经定义好了，可以直接传入文本并返回结果即可。

```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """
    生成输入文本列表的 embedding.
    Args:
        texts (List[str]): 要生成 embedding 的文本列表.

    Returns:
        List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
    """
    return [self._embed(text) for text in texts]
```



`embed_query` 是对单个文本计算 embedding 的方法，因为我们已经定义好对文档列表计算 embedding 的方法`embed_documents` 了，这里可以直接将单个文本组装成 list 的形式传给 `embed_documents`。

如果文档特别长，我们可以考虑对文档分段，防止超过最大 token 限制。
