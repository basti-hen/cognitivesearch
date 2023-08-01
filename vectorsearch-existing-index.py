import openai
from dotenv import load_dotenv
import time
import os 
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    Similarity,
    SemanticSettings,
    SemanticConfiguration,
    SemanticField,
    PrioritizedFields
)
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential


# Configure environment variables
load_dotenv()
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
credential = AzureKeyCredential(key)

# Get the existing index
index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
index = index_client.get_index("realestate-us-sample-index")

field = SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
    searchable=True, vector_search_dimensions=1536, vector_search_configuration="my-vector-config")


vector_search = VectorSearch(
    algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(
            name="my-vector-config",
            kind="hnsw",
            hnsw_parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 500,
                "metric": "cosine"
            }
        )
    ]
)
# Check if the content vector is already in the index
if field.name in [f.name for f in index.fields]:
    print("Content vector already in index, skipping update")
else:
    index.fields.append(field)

    # Get all suggesters from the index (if the index has suggestors in it, the update would fail.)
    suggesters = index.suggesters

    # Update the index in Azure Search

    index = SearchIndex(name=index_name, suggesters=suggesters, fields=index.fields,
                        vector_search=vector_search)
    result = index_client.create_or_update_index(index)
    print(f'Index {result.name} updated')



# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text):
    retries = 3
    wait_time = 60  # seconds
    for i in range(retries):
        try:
            response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
            embedding = response['data'][0]['embedding']
            return embedding
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}. Retrying in {wait_time} seconds...")
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}. Retrying in {wait_time} seconds...")
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}. Retrying in {wait_time} seconds...")
        time.sleep(wait_time)
    raise Exception(f"Request failed after {retries} retries")
    

# Select field names in Cognitive Searh
primary_key = 'listingId'
field_to_vectorize = 'description'

# Get the documents from the search index
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
results = search_client.search(search_text="*",select=[primary_key, field_to_vectorize])  


# Generate embeddings for each document
for result in results:
    print(f"Result {result}")
    description = result['description']
    embeddings = generate_embeddings(description)
    document = {
        "@search.action": "merge",
        primary_key: result[primary_key],
        "contentVector": embeddings
    }
    search_client.upload_documents(documents=[document])

