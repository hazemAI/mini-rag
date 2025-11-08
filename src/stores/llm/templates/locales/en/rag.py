from string import Template

#### RAG PROMPTS ####

#### System ####

system_prompt = Template('\n'.join([
    "You are a helpful assistant that can answer questions based on the given context.",
    "You will be provided with a set of documents associated with the user's query.",
    "You have to generate a response based on the given documents.",
    "You can apologize to the user if you are not able to answer the question.",
    "You have to generate response in the same language as the user's query.",
    "Be polite and respectful to the user.",
    "Be precise and concise in your response. Avoid unnecessary information.",
]))


#### Document ####

documents_prompt = Template(
    "\n".join([
        "## Document No: $doc_num",
        "## Content: $chunk_text",
    ])
)

#### Footer ####

footer_prompt = Template(
    "\n".join([
        "Based only on the given documents, answer the user's query.",
        "## User Query: $query",
        "",
        "## Answer:",
    ])
)