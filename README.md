# Large-language-model-project
## 介绍
在这个项目中，我们尝试使用大预言模型进行文本分类和命名实体识别。通常来说，完成文本分类和命名实体识别任务一般使用深度学习来完成，简单又快捷。但是开源大语言模型越来越多，或许我们可以尝试使用大语言模型来完成，其实我是想做大语言模型的多任务处理，我想试一试通过大语言模型同时处理多个任务会怎么样。在这里我选择分类和命名实体识别任务。  

因为我的专业是食品方向的，所以我选择用食品网络舆情数据来做。我的思路也很简单，首先判断这个文本是否属于“食品安全事件”，如果是，再把涉及到的食品种类、地点、组织提取出来。

## 大模型
本实验使用到的模型有[ChatGLM4-9B](https://github.com/THUDM/GLM-4)    [Qwen2.5-14B](https://github.com/QwenLM/Qwen3)    [Qwen2.5-0.5B](https://github.com/QwenLM/Qwen3)    [Llama3.1-8B](https://github.com/meta-llama/llama3)  

模型下载地址: [ChatGLM4-9B](https://huggingface.co/THUDM/glm-4-9b)    [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B)    [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)    [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)

就在前几天，Qwen3系列问世，建议大家尝试Qwen3系列的模型，或许会有意想不到的效果

## 模型微调
在这里我用的是[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，LLama—Factory对于新手来说非常友好，并且支持市面上常见的各种模型，也支持图形化界面微调

# 持续更新中。。。
