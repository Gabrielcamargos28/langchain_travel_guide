from openai import OpenAI

client = OpenAI()

descricao_quadrinho = "Aventuras épicas no espaço com heróis e vilões"

response = client.embeddings.create(
    input=descricao_quadrinho,
    model="text-embedding-3-small",
)

print(response.data)