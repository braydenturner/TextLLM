# TextLLM


## Getting Text Data

Found the `chat.db` database locally at `'/Users/{user}/Library/Messages/`

And address book at `/Users/{user}/Library/Application Support/AddressBook/Sources/247C28AF-9246-4320-ABB0-83EA82ABAA52/AddressBook-v22.abcddb`


SQL was then used to extract a bunch of different tables in to pabndas dataframes and merged in to one table


## Data Analysis
The `seaborn` library was used for visuals


## Embeddings

A few different models were attempted. A few classes were setup to swap between them

```python
class Embedding:

    class Type(Enum):
        DEFAULT = 1
        OPENAI = 2
        SENTENCETRANSFORMER = 3
    
    def __init__(self):
        pass
    
    @classmethod
    def model(cls, type: Type = Type.DEFAULT):
        match type:
            case Embedding.Type.DEFAULT:
                return DefaultEmbedding()
            case Embedding.Type.OPENAI:
                return OpenAIEmbedding()
            case Embedding.Type.SENTENCETRANSFORMER:
                return SentenceTransformerEmbedding()
            
        raise TypeError
    
    def embed(self):
        pass

class OpenAIEmbedding(Embedding):
    
    def __init__(self):
        super(OpenAIEmbedding, self)
        self.client = OpenAI(api_key=personal_api_key)
        self.model = "text-embedding-3-small"
            
    def embed(self, text):
        return self.client.embeddings.create(
            input=[text.replace('\n', ' ')],
            model=self.model).data[0].embedding

class DefaultEmbedding(Embedding):

    def __init__(self):
        super(DefaultEmbedding, self)
        self.model = embedding_functions.DefaultEmbeddingFunction()

    def embed(self, text):
        return self.model(text)[0]

class SentenceTransformerEmbedding(Embedding):

    def __init__(self):
        super(SentenceTransformerEmbedding, self)
        self.model =  embedding_functions.SentenceTransformerEmbeddingFunction('paraphrase-MiniLM-L6-v2')

    def embed(self, text):
        return self.model(text)[0]
```


* OpenAIEmbedding - Uses the model form OpenAI
* DefaultEmbedding - Default embedding function used by the chromadb library
* SentenceTransformerEmbedding - Uses the open source hugging face model

## Vector Database
ChromaDB was used for storing and retrieving nearest embeddings. To not exceed insert limit, batches were needed

```python
# Batches needed to not exceed insert limit
batches = create_batches(api=client,ids=list(ids), documents=list(documents), embeddings=embeddings, metadatas=metadatas)
for batch in batches:
    print(f"Adding batch of size {len(batch[0])}")
    collection.add(ids=batch[0],
                   documents=batch[3],
                   metadatas=batch[2],
                   embeddings=batch[1])
```


## Model with Tools
To be continued