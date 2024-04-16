## 一些基本概念

### Prompt是什么

prompt可以是一个简单的句子、问题或者命令，它告诉模型要执行什么任务生成什么类型的响应，比如我们需要一个摘要文档的模型，那我们的prompt可以是这样：

```tex
请为下面文章生成一个摘要：
```

然后跟上自己希望模型摘要的内容，模型会根据这个prompt来生成摘要。

### Temperature

LLM的生成是具有随机性的，控制temperature参数的大小来控制LLM生成结果的随机性和创造性。

Temperature 一般取值在 0~1 之间，当取值较低接近 0 时，预测的随机性会较低，产生更保守、可预测的文本，不太可能生成意想不到或不寻常的词。当取值较高接近 1 时，预测的随机性会较高，所有词被选择的可能性更大，会产生更有创意、多样化的文本，更有可能生成不寻常或意想不到的词。

我们一般将 temperature 设置为 0，从而保证助手对知识库内容的稳定使用，规避错误内容、模型幻觉；在产品智能客服、科研论文写作等场景中，我们同样更需要稳定性而不是创造性；但在个性化 AI、创意营销文案生成等场景中，我们就更需要创意性，从而更倾向于将 temperature 设置为较高的值。

### System Prompt

你可以设置两种 Prompt：一种是 System Prompt，该种 Prompt 内容会在整个会话过程中持久地影响模型的回复，且相比于普通 Prompt 具有更高的重要性；另一种是 User Prompt，这更偏向于我们平时提到的 Prompt，即需要模型做出回复的输入。

```json
{
    "system prompt": "你是一个幽默风趣的个人知识库助手，可以根据给定的知识库内容回答用户的提问，注意，你的回答风格应是幽默风趣的",
    "user prompt": "我今天有什么事务？"
}
```
