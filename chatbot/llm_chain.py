from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answers(inputs, llm):
    question = inputs["question"]
    docs = inputs["docs"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke({
                    "question": question,
                    "context": doc.page_content
                }).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"]
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)



def choose_answer(inputs, llm):
    question = inputs["question"]
    answers = inputs["answers"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{a['answer']}\nSource:{a['source']}\nDate:{a['date']}" for a in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})
