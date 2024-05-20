# This script can run both locally (w/LM Studio) or with an OpenAI key.
from openai import OpenAI
import numpy as np
import json
from config import *

embeddings_json= "knowledge_pool/Cubic.json"

# Choose between "local" or "openai" mode
mode = "local" # or "local"
client, completion_model = api_mode(mode)

# question = "Công ty cubic hoạt động trong lĩnh vực nào"
# question = "Một Dấu ấn quan trọng nhất của công ty Cubic là gì?"
question = "Cubic có BIM không"
# question = "công ty con của Cubic tên là gì"


num_results = 3 #how many vectors to retrieve


def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    response = local_client.embeddings.create(input = [text], model=model)
    vector = response.data[0].embedding
    return vector

def similarity(v1, v2):
    return np.dot(v1, v2)

def load_embeddings(embeddings_json):
    with open(embeddings_json, 'r', encoding='utf8') as infile:
        return json.load(infile)
    
def get_vectors(question_vector, index_lib):
    scores = []
    for vector in index_lib:
        score = similarity(question_vector, vector['vector'])
        scores.append({'content': vector['content'], 'score': score})

    scores.sort(key=lambda x: x['score'], reverse=True)
    best_vectors = scores[0:num_results]
    return best_vectors

def rag_answer(question, prompt, model=completion_model[0]["model"]):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", 
             "content": prompt
            },
            {"role": "user", 
             "content": question
            }
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content
print("="*50)
print("CÂU HỎI:")
print(question)
print("...Đang chờ câu trả lời...")
# Embed our question
question_vector = get_embedding(question)

# Load the knowledge embeddings
index_lib = load_embeddings(embeddings_json)

# Retrieve the best vectors
scored_vectors = get_vectors(question_vector,index_lib)
scored_contents = [vector['content'] for vector in scored_vectors]
rag_result = "\n".join(scored_contents)

# Get answer from rag informed agent
prompt = f"""Trả lời câu hỏi dựa trên thông tin được cung cấp.
             Bạn được cung cấp các phần được trích xuất của một tài liệu dài và một câu hỏi. Đưa ra câu trả lời trực tiếp.
             Nếu bạn không biết câu trả lời, chỉ cần nói "Tôi không biết". Đừng bịa ra một câu trả lời
            PROVIDED INFORMATION: """ + rag_result

answer = rag_answer(question, prompt)

# print(prompt)
# print("-"*50)
# print("RAG_RESULT:")
# print(rag_result)
print("-"*50)
print("CUBIC AI:")
print(answer)
print("="*50)
