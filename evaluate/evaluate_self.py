import os
import sys

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from milvus_db.milvus_retrieve import MilvusRetriever
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import List, Dict
from llm_utils import llm, qwen_embeddings
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextPrecisionWithReference
from env_utils import COLLECTION_NAME, MILVUS_URI
from pymilvus import MilvusClient
import asyncio


def generate_answer(question: str, contexts: List[Dict]) -> str:
    """
    利用llm根据上下文生成答案

    Args:
        question: 问题
        contexts: 检索到的上下文列表 (包含text category 等字段)

    Returns:
        str: 答案
    """
    # 将检索到的上下文格式化为字符串，便于LLM理解
    # 每个上下文前加上"上下文x"的标识，方便LLM区分
    context_str = "\n\n".join([f"上下文{i+1}: {context['text']}" for i, context in enumerate(contexts)])

    # 提示词模板
    prompt = f"""
    你是一位AI助手,请根据提供的上下文来回答用户的问题。请保证回答一定是基于提供的上下文，不要编造信息。

    用户问题: {question}

    上下文: {context_str}

    请给予以上上下文回答用户提供的问题，可以带有表情符号。
    """

    # 调用LLM生成答案
    response = llm.invoke(prompt)
    return response.content

class RAGEvaluator:
    """
    RAG 评估类
    :params evaluator_llm 利用llm做推理评估
    :params evaluator_embedding 利用embedding模型做语义评估
    """
    def __init__(self, evaluator_llm, evaluator_embedding):
        self.evaluator_llm = evaluator_llm
        self.evaluator_embedding = evaluator_embedding

    async def evaluate_metrics(self, question: str, contexts: List[Dict], response: str, reference: str = None):
        """
        评估RAG模型的指标:
        1.无参考reference的上下文精确度 LLMContextPrecisionWithoutReference 指标可在无参考答案的情况下使用。该方法通过 LLM 比较 retrieved_contexts 中的每个片段与 response ，以评估检索到的上下文是否相关。
        2.有参考reference的上下文精确度 当您同时拥有检索到的上下文和与查询相关联的参考响应时，可以使用该指标。为了评估检索到的上下文是否相关，该方法使用 LLM 将检索到的上下文中的每个块与参考响应进行比较。

        Args:
            question: 问题
            contexts: 检索到的上下文列表 (包含text category 等字段)
            response: RAG模型生成的答案
            reference: 参考答案(用于评估的基准答案，通常为已知的正确答案) 可选

        Returns:
            Dict: 评估指标
        """
        # 1.创建评估样本SingleTurnSample
        sample = SingleTurnSample(
            user_input=question,          # 使用 user_input 而不是 question
            retrieved_contexts=[f"上下文{i+1}: {context['text']}" for i, context in enumerate(contexts)],    # 检索到的上下文 text 字段是我们需要的
            response=response,            # RAG模型生成的答案
            reference=reference           # 参考答案(用于评估的基准答案，通常为已知的正确答案) 可选
        )
        
        # 2.创建精确度评估指标
        if reference:
            # 有参考答案
            context_precision = LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        else:
            # 无参考答案
            context_precision = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
        
        # 3.评估
        context_precision_score = await context_precision.single_turn_ascore(sample)
        print(f"上下文精确度评估指标: {context_precision_score}")
    

async def main():
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embedding = LangchainEmbeddingsWrapper(qwen_embeddings)

    # 创建RAG evaluator
    rag_evaluator = RAGEvaluator(evaluator_llm, evaluator_embedding)

    question = "In the Contamination data for exam AP Physics 2,GPT-4 (no vision) and GPT-4 get what?"
    # 检索上下文，从Milvus数据库获取
    m_re = MilvusRetriever(collection_name=COLLECTION_NAME, milvus_client=MilvusClient(uri=MILVUS_URI, user='root', password='Milvus'))
    contexts = m_re.retrieve(question)

    generated_answer = generate_answer(question, contexts)
    await rag_evaluator.evaluate_metrics(question, contexts, generated_answer)

    print(f"{'\n'.join([f'上下文{i+1}: {context["text"]}' for i, context in enumerate(contexts)])}")
    print('*'*100)
    print(f"生成答案: {generated_answer}")   
    print('*'*100)

if __name__ == "__main__":
    asyncio.run(main())