## 如何评估LLM应用

### 人工评估

**量化评估**

修改提示词（prompt），对模型的回答进行人工打分。

**多维评估**

一个优秀的问答助手，应当既能够很好地回答用户的问题，保证答案的正确性，又能够体现出充分的智能性。

设计每个维度的评估指标，在每个维度上都进行打分，从而综合评估系统性能。

例如，在本项目中，我们可以设计如下几个维度的评估：

```
① 知识查找正确性。该维度需要查看系统从向量数据库查找相关知识片段的中间结果，评估系统查找到的知识片段是否能够对问题做出回答。该维度为0-1评估，即打分为0指查找到的知识片段不能做出回答，打分为1指查找到的知识片段可以做出回答。

② 回答一致性。该维度评估系统的回答是否针对用户问题展开，是否有偏题、错误理解题意的情况，该维度量纲同样设计为0~1，0为完全偏题，1为完全切题，中间结果可以任取。

③ 回答幻觉比例。该维度需要综合系统回答与查找到的知识片段，评估系统的回答是否出现幻觉，幻觉比例有多高。该维度同样设计为0~1,0为全部是模型幻觉，1为没有任何幻觉。

④ 回答正确性。该维度评估系统回答是否正确，是否充分解答了用户问题，是系统最核心的评估指标之一。该维度可以在0~1之间任意打分。

① 知识查找正确性——1

② 回答一致性——0.8（解答了问题，但是类似于“反馈”的话题偏题了）

③ 回答幻觉比例——1

④ 回答正确性——0.8（理由同上）

⑤ 逻辑性——0.7（后续内容与前面逻辑连贯性不强）

⑥ 通顺性——0.6（最后总结啰嗦且无效）

⑦ 智能性——0.5（具有 AI 回答的显著风格）
```



### 简单自动评估

由于大模型的不稳定性，即使我们要求其只给出选择选项，系统可能也会返回一大堆文字，其中详细解释了为什么选择如下选项。因此，我们需要将选项从模型回答中抽取出来。同时，我们需要设计一个打分策略。一般情况下，我们可以使用多选题的一般打分策略：全选1分，漏选0.5分，错选不选不得分：

```
def multi_select_score_v1(true_answer : str, generate_answer : str) -> float:
    # true_anser : 正确答案，str 类型，例如 'BCD'
    # generate_answer : 模型生成答案，str 类型
    true_answers = list(true_answer)
    '''为便于计算，我们假设每道题都只有 A B C D 四个选项'''
    # 先找出错误答案集合
    false_answers = [item for item in ['A', 'B', 'C', 'D'] if item not in true_answers]
    # 如果生成答案出现了错误答案
    for one_answer in false_answers:
        if one_answer in generate_answer:
            return 0
    # 再判断是否全选了正确答案
    if_correct = 0
    for one_answer in true_answers:
        if one_answer in generate_answer:
            if_correct += 1
            continue
    if if_correct == 0:
        # 不选
        return 0
    elif if_correct == len(true_answers):
        # 全选
        return 1
    else:
        # 漏选
        return 0.5
```

基于上述打分函数，我们可以测试四个回答：

① B C

② 除了 A 周志华之外，其他都是南瓜书的作者

③ 应该选择 B C D

④ 我不知道

```
answer1 = 'B C'
answer2 = '西瓜书的作者是 A 周志华'
answer3 = '应该选择 B C D'
answer4 = '我不知道'
true_answer = 'BCD'
print("答案一得分：", multi_select_score_v1(true_answer, answer1))
print("答案二得分：", multi_select_score_v1(true_answer, answer2))
print("答案三得分：", multi_select_score_v1(true_answer, answer3))
print("答案四得分：", multi_select_score_v1(true_answer, answer4))
答案一得分： 0.5
答案二得分： 0
答案三得分： 1
答案四得分： 0
```

但是我们可以看到，我们要求模型在不能回答的情况下不做选择，而不是随便选。但是在我们的打分策略中，错选和不选均为0分，这样其实鼓励了模型的幻觉回答，因此我们可以根据情况调整打分策略，让错选扣一分：

```
def multi_select_score_v2(true_answer : str, generate_answer : str) -> float:
    # true_anser : 正确答案，str 类型，例如 'BCD'
    # generate_answer : 模型生成答案，str 类型
    true_answers = list(true_answer)
    '''为便于计算，我们假设每道题都只有 A B C D 四个选项'''
    # 先找出错误答案集合
    false_answers = [item for item in ['A', 'B', 'C', 'D'] if item not in true_answers]
    # 如果生成答案出现了错误答案
    for one_answer in false_answers:
        if one_answer in generate_answer:
            return -1
    # 再判断是否全选了正确答案
    if_correct = 0
    for one_answer in true_answers:
        if one_answer in generate_answer:
            if_correct += 1
            continue
    if if_correct == 0:
        # 不选
        return 0
    elif if_correct == len(true_answers):
        # 全选
        return 1
    else:
        # 漏选
        return 0.5
```

如上，我们使用第二版本的打分函数再次对四个答案打分：

```
answer1 = 'B C'
answer2 = '西瓜书的作者是 A 周志华'
answer3 = '应该选择 B C D'
answer4 = '我不知道'
true_answer = 'BCD'
print("答案一得分：", multi_select_score_v2(true_answer, answer1))
print("答案二得分：", multi_select_score_v2(true_answer, answer2))
print("答案三得分：", multi_select_score_v2(true_answer, answer3))
print("答案四得分：", multi_select_score_v2(true_answer, answer4))
答案一得分： 0.5
答案二得分： -1
答案三得分： 1
答案四得分： 0
```

可以看到，这样我们就实现了快速、自动又有区分度的自动评估。在这样的方法下，我们只需对每一个验证案例进行构造，之后每一次验证、迭代都可以完全自动化进行，从而实现了高效的验证。

### 相似度比较

BLEU: a Method for Automatic Evaluation of Machine Translation，相似度比较算法

我们可以调用 nltk 库中的 bleu 打分函数来计算：

```
from nltk.translate.bleu_score import sentence_bleu
import jieba

def bleu_score(true_answer : str, generate_answer : str) -> float:
    # true_anser : 标准答案，str 类型
    # generate_answer : 模型生成答案，str 类型
    true_answers = list(jieba.cut(true_answer))
    # print(true_answers)
    generate_answers = list(jieba.cut(generate_answer))
    # print(generate_answers)
    bleu_score = sentence_bleu(true_answers, generate_answers)
    return bleu_score
```

测试一下：

```
true_answer = '周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充具体的推导细节。'

print("答案一：")
answer1 = '周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充具体的推导细节。'
print(answer1)
score = bleu_score(true_answer, answer1)
print("得分：", score)
print("答案二：")
answer2 = '本南瓜书只能算是我等数学渣渣在自学的时候记下来的笔记，希望能够帮助大家都成为一名合格的“理工科数学基础扎实点的大二下学生”'
print(answer2)
score = bleu_score(true_answer, answer2)
print("得分：", score)
答案一：
周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充具体的推导细节。
得分： 1.2705543769116016e-231
答案二：
本南瓜书只能算是我等数学渣渣在自学的时候记下来的笔记，希望能够帮助大家都成为一名合格的“理工科数学基础扎实点的大二下学生”
得分： 1.1935398790363042e-231
```

可以看到，答案与标准答案一致性越高，则评估打分就越高。

但是，该种方法同样存在**几个问题**：

① 人工构造标准答案可能是一件困难的事情

②生成回答与标准答案高度一致但在核心的几个地方恰恰相反导致答案完全错误，bleu 得分仍然会很高

③灵活性很差，如果模型生成了比标准答案更好的回答，但评估得分反而会降低

④ 无法评估回答的智能性、流畅性。如果回答是各个标准答案中的关键词拼接出来的，我们认为这样的回答是不可用无法理解的，但 bleu 得分会较高

### 大模型辅助评估

通过构造 Prompt Engineering 让大模型充当一个评估者的角色，从而替代人工评估的评估员；

我们可以构造如下的 Prompt Engineering，让大模型进行打分：

```
prompt = '''
你是一个模型回答评估员。
接下来，我将给你一个问题、对应的知识片段以及模型根据知识片段对问题的回答。
请你依次评估以下维度模型回答的表现，分别给出打分：

① 知识查找正确性。评估系统给定的知识片段是否能够对问题做出回答。如果知识片段不能做出回答，打分为0；如果知识片段可以做出回答，打分为1。

② 回答一致性。评估系统的回答是否针对用户问题展开，是否有偏题、错误理解题意的情况，打分分值在0~1之间，0为完全偏题，1为完全切题。

③ 回答幻觉比例。该维度需要综合系统回答与查找到的知识片段，评估系统的回答是否出现幻觉，打分分值在0~1之间,0为全部是模型幻觉，1为没有任何幻觉。

④ 回答正确性。该维度评估系统回答是否正确，是否充分解答了用户问题，打分分值在0~1之间，0为完全不正确，1为完全正确。

⑤ 逻辑性。该维度评估系统回答是否逻辑连贯，是否出现前后冲突、逻辑混乱的情况。打分分值在0~1之间，0为逻辑完全混乱，1为完全没有逻辑问题。

⑥ 通顺性。该维度评估系统回答是否通顺、合乎语法。打分分值在0~1之间，0为语句完全不通顺，1为语句完全通顺没有任何语法问题。

⑦ 智能性。该维度评估系统回答是否拟人化、智能化，是否能充分让用户混淆人工回答与智能回答。打分分值在0~1之间，0为非常明显的模型回答，1为与人工回答高度一致。

你应该是比较严苛的评估员，很少给出满分的高评估。
用户问题：
~~~
{}
~~~
待评估的回答：
~~~
{}
~~~
给定的知识片段：
~~~
{}
~~~
你应该返回给我一个可直接解析的 Python 字典，字典的键是如上维度，值是每一个维度对应的评估打分。
不要输出任何其他内容。
'''
```

我们可以实际测试一下其效果：

```
# 使用第二章讲过的 OpenAI 原生接口

from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def gen_gpt_messages(prompt):
    '''
    构造 GPT 模型请求参数 messages
    
    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


def get_completion(prompt, model="gpt-3.5-turbo", temperature = 0):
    '''
    获取 GPT 模型调用结果

    请求参数：
        prompt: 对应的提示词
        model: 调用的模型，默认为 gpt-3.5-turbo，也可以按需选择 gpt-4 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
    '''
    response = client.chat.completions.create(
        model=model,
        messages=gen_gpt_messages(prompt),
        temperature=temperature,
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

question = "应该如何使用南瓜书？"
result = qa_chain({"query": question})
answer = result["result"]
knowledge = result["source_documents"]

response = get_completion(prompt.format(question, answer, knowledge))
response
'{\n    "知识查找正确性": 1,\n    "回答一致性": 0.9,\n    "回答幻觉比例": 0.9,\n    "回答正确性": 0.9,\n    "逻辑性": 0.9,\n    "通顺性": 0.9,\n    "智能性": 0.8\n}'
```

## 评估和优化生成部分(Agent思想)

1. 修改prompt内容，提高生成内容的质量

2. 标明回答的来源，在编写template的时候，要求附上回答的原文的来源。

3. **减少模型的幻觉**：优化 Prompt，将之前的 Prompt 变成两个步骤，要求模型在第二个步骤中做出反思

   ```python
   template_v4 = """
   请你依次执行以下步骤：
   ① 使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
   你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。
   如果答案有几点，你应该分点标号回答，让答案清晰具体。
   上下文：
   {context}
   问题: 
   {question}
   有用的回答:
   ② 基于提供的上下文，反思回答中有没有不正确或不是基于上下文得到的内容，如果有，回答你不知道
   确保你执行了每一个步骤，不要跳过任意一个步骤。
   """
   
   QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template_v4)
   qa_chain = RetrievalQA.from_chain_type(llm,
                                          retriever=vectordb.as_retriever(),
                                          return_source_documents=True,
                                          chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
   
   question = "我们应该如何去构造一个LLM项目"
   result = qa_chain({"query": question})
   print(result["result"])
   
   # 根据回答可以看出，要求模型做出自我反思之后，模型修复了自己的幻觉，给出了正确的答案
   ```

4. 我们需要模型以我们指定的格式进行输出。但是，由于我们使用了 Prompt Template 来填充用户问题，用户问题中存在的格式要求往往会被忽略，针对该问题，一个存在的解决方案是，在我们的检索 LLM 之前，增加一层 LLM 来实现指令的解析，将用户问题的格式要求和问题内容拆分开来。针对用户指令，设置一个 LLM（即 Agent）来理解指令，判断指令需要执行什么工具，再针对性调用需要执行的工具，其中每一个工具可以是基于不同 Prompt Engineering 的 LLM，也可以是例如数据库、API 等。

   ```python
   from openai import OpenAI
   
   client = OpenAI(
       # This is the default and can be omitted
       api_key=os.environ.get("OPENAI_API_KEY"),
   )
   
   
   def gen_gpt_messages(prompt):
       '''
       构造 GPT 模型请求参数 messages
       
       请求参数：
           prompt: 对应的用户提示词
       '''
       messages = [{"role": "user", "content": prompt}]
       return messages
   
   
   def get_completion(prompt, model="gpt-3.5-turbo", temperature = 0):
       '''
       获取 GPT 模型调用结果
   
       请求参数：
           prompt: 对应的提示词
           model: 调用的模型，默认为 gpt-3.5-turbo，也可以按需选择 gpt-4 等其他模型
           temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
       '''
       response = client.chat.completions.create(
           model=model,
           messages=gen_gpt_messages(prompt),
           temperature=temperature,
       )
       if len(response.choices) > 0:
           return response.choices[0].message.content
       return "generate answer error"
   
   prompt_input = '''
   请判断以下问题中是否包含对输出的格式要求，并按以下要求输出：
   请返回给我一个可解析的Python列表，列表第一个元素是对输出的格式要求，应该是一个指令；第二个元素是去掉格式要求的问题原文
   如果没有格式要求，请将第一个元素置为空
   需要判断的问题：
   ~~~
   {}
   ~~~
   不要输出任何其他内容或格式，确保返回结果可解析。
   '''
   ```

   我们测试一下该 LLM 分解格式要求的能力：

   ```python
   response = get_completion(prompt_input.format(question))
   response
   '```\n["给我返回一个 Python List", "LLM的分类是什么？"]\n```'
   ```

   可以看到，通过上述 Prompt，LLM 可以很好地实现输出格式的解析，接下来，我们可以再设置一个 LLM 根据输出格式要求，对输出内容进行解析：

   ```python
   prompt_output = '''
   请根据回答文本和输出格式要求，按照给定的格式要求对问题做出回答
   需要回答的问题：
   ~~~
   {}
   ~~~
   回答文本：
   ~~~
   {}
   ~~~
   输出格式要求：
   ~~~
   {}
   ~~~
   '''
   ```

   然后我们可以将两个 LLM 与检索链串联起来：

   ```python
   question = 'LLM的分类是什么？给我返回一个 Python List'
   # 首先将格式要求与问题拆分
   input_lst_s = get_completion(prompt_input.format(question))
   # 找到拆分之后列表的起始和结束字符
   start_loc = input_lst_s.find('[')
   end_loc = input_lst_s.find(']')
   rule, new_question = eval(input_lst_s[start_loc:end_loc+1])
   # 接着使用拆分后的问题调用检索链
   result = qa_chain({"query": new_question})
   result_context = result["result"]
   # 接着调用输出格式解析
   response = get_completion(prompt_output.format(new_question, result_context, rule))
   response
   "['基础LLM', '指令微调LLM']"
   ```

## 评估并优化检索部分

生成的前提是检索，只有当我们应用的检索部分能够根据用户 query 检索到正确的答案文档时，大模型的生成结果才可能是正确的。因此，检索部分的检索精确率和召回率其实更大程度影响了应用的整体性能。

![image-20240425234224809](C:\Users\asd61\AppData\Roaming\Typora\typora-user-images\image-20240425234224809.png)



针对用户输入的一个 query，系统会将其转化为向量并在向量数据库中匹配最相关的文本段，然后根据我们的设定选择 3～5 个文本段落和用户的 query 一起交给大模型，再由大模型根据检索到的文本段落回答用户 query 中提出的问题。在这一整个系统中，我们将向量数据库检索相关文本段落的部分称为检索部分，将大模型根据检索到的文本段落进行答案生成的部分称为生成部分。

因此，检索部分的核心功能是找到存在于知识库中、能够正确回答用户 query 中的提问的文本段落。因此，我们可以定义一个最直观的准确率在评估检索效果：对于 N 个给定 query，我们保证每一个 query 对应的正确答案都存在于知识库中。假设对于每一个 query，系统找到了 K 个文本片段，如果正确答案在 K 个文本片段之一，那么我们认为检索成功；如果正确答案不在 K 个文本片段之一，我们任务检索失败。那么，系统的检索准确率可以被简单地计算为：



accuracy=𝑀/𝑁



其中，M 是成功检索的 query 数。



在优化检索效果时，以下几个关键点非常重要：

###  1. 知识片段被割裂导致答案丢失

该问题一般表现为，对于一个用户 query，我们可以确定其问题一定是存在于知识库之中的，但是我们发现检索到的知识片段将正确答案分割开了，导致不能形成一个完整、合理的答案。该种问题在需要较长回答的 query 上较为常见。

该类问题的一般优化思路是，优化文本切割方式。根据特定字符和 chunk 大小进行分割，但该类分割方式往往不能照顾到文本语义，容易造成同一主题的强相关上下文被切分到两个 chunk 总。对于一些格式统一、组织清晰的知识文档，我们可以针对性构建更合适的分割规则；对于格式混乱、无法形成统一的分割规则的文档，我们可以考虑纳入一定的人力进行分割。我们也可以考虑训练一个专用于文本分割的模型，来实现根据语义和主题的 chunk 切分。

### 2. query 提问需要长上下文概括回答

该问题也是存在于知识库构建的一个问题。即部分 query 提出的问题需要检索部分跨越很长的上下文来做出概括性回答，也就是需要跨越多个 chunk 来综合回答问题。但是由于模型上下文限制，我们往往很难给出足够的 chunk 数。

该类问题的一般优化思路是，优化知识库构建方式。针对可能需要此类回答的文档，我们可以增加一个步骤，通过使用 LLM 来对长文档进行概括总结，或者预设提问让 LLM 做出回答，从而将此类问题的可能答案预先填入知识库作为单独的 chunk，来一定程度解决该问题。

### 3. 关键词误导

该问题一般表现为，对于一个用户 query，系统检索到的知识片段有很多与 query 强相关的关键词，但知识片段本身并非针对 query 做出的回答。这种情况一般源于 query 中有多个关键词，其中次要关键词的匹配效果影响了主要关键词。

该类问题的一般优化思路是，对用户 query 进行改写，这也是目前很多大模型应用的常用思路。即对于用户输入 query，我们首先通过 LLM 来将用户 query 改写成一种合理的形式，去除次要关键词以及可能出现的错字、漏字的影响。具体改写成什么形式根据具体业务而定，可以要求 LLM 对 query 进行提炼形成 Json 对象，也可以要求 LLM 对 query 进行扩写等。

### 4. 匹配关系不合理

该问题是较为常见的，即匹配到的强相关文本段并没有包含答案文本。该问题的核心问题在于，我们使用的向量模型和我们一开始的假设不符。在讲解 RAG 的框架时，我们有提到，RAG 起效果是有一个核心假设的，即我们假设我们匹配到的强相关文本段就是问题对应的答案文本段。但是很多向量模型其实构建的是“配对”的语义相似度而非“因果”的语义相似度，例如对于 query-“今天天气怎么样”，会认为“我想知道今天天气”的相关性比“天气不错”更高。

该类问题的一般优化思路是，优化向量模型或是构建倒排索引。我们可以选择效果更好的向量模型，或是收集部分数据，在自己的业务上微调一个更符合自己业务的向量模型。我们也可以考虑构建倒排索引，即针对知识库的每一个知识片段，构建一个能够表征该片段内容但和 query 的相对相关性更准确的索引，在检索时匹配索引和 query 的相关性而不是全文，从而提高匹配关系的准确性。

## Prompt构建准则

```
1. 编写清晰、具体的指令是构造 Prompt 的第一原则。Prompt需要明确表达需求，提供充足上下文，使语言模型准确理解意图。过于简略的Prompt会使模型难以完成任务。

2. 给予模型充足思考时间是构造Prompt的第二原则。语言模型需要时间推理和解决复杂问题，匆忙得出的结论可能不准确。因此，Prompt应该包含逐步推理的要求，让模型有足够时间思考，生成更准确的结果。

3. 在设计Prompt时，要指定完成任务所需的步骤。通过给定一个复杂任务，给出完成任务的一系列步骤，可以帮助模型更好地理解任务要求，提高任务完成的效率。

4. 迭代优化是构造Prompt的常用策略。通过不断尝试、分析结果、改进Prompt的过程，逐步逼近最优的Prompt形式。成功的Prompt通常是通过多轮调整得出的。

5. 添加表格描述是优化Prompt的一种方法。要求模型抽取信息并组织成表格，指定表格的列、表名和格式，可以帮助模型更好地理解任务，并生成符合预期的结果。

总之，构造Prompt的原则包括清晰具体的指令、给予模型充足思考时间、指定完成任务所需的步骤、迭代优化和添加表格描述等。这些原则可以帮助开发者设计出高效、可靠的Prompt，发挥语言模型的最大潜力。
```
