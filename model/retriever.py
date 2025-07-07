from faiss import IDSelectorArray, SearchParameters
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

import spacy
import faiss

# Load the English model
nlp = spacy.load("en_core_web_sm")

class Retriever:
    """
    Handles the retrieval of relevant documents from a pre-built FAISS index.
    Enables querying with sentence transformers embeddings.

    Attributes:
        index (faiss.Index): FAISS index for fast similarity search.
        doc_info (pd.DataFrame): DataFrame containing detailed information about documents.
        documents (list of str): List of original documents.
        embedding_model (SentenceTransformer): Model used for embedding the documents and queries.
    """

    def __init__(self, index, doc_info, embedding_model_name):
        """Initializes the Retriever class with necessary components.

        Args:
            index: FAISS index for fast retrieval.
            doc_info (DataFrame): DataFrame containing info about embedded document; aligned indices with index embeddings.
            documents (list): List of original documents.
            embedding_model_name (str): Name of the sentence transformer model.
        """
        self.index = index
        self.doc_info = doc_info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)

    def retrieve(self, query_batch, k):
        """
        Retrieves the top-k most similar documents for each query in a batch of queries.

        Args:
            query_batch (list of str): List of query strings.
            k (int): Number of documents to retrieve.

        Returns:
            List[List[dict]]: List of lists containing formatted results of retrieved documents for each query.
        """
        # Batch encode the queries
        query_embeddings = self.embedding_model.encode(query_batch, show_progress_bar=False)

        # Process each query separately
        results_batch = []
        for query_embedding in query_embeddings:
            # Search the index for similar documents, retrieve a larger set of documents
            similarities, indices = self.index.search(np.array([query_embedding]), k)

            # Convert 2D arrays into 1D arrays
            indices, similarities = indices[0], similarities[0]
           
            # Get the first and the last documents
            results_batch.append([self._create_result(indices[0], similarities[0])])
            results_batch.append([self._create_result(indices[len(indices)-1], similarities[len(similarities)-1])])

        return results_batch


    def _create_result(self, idx, score):
        """
        Creates/builds a result dictionary of the retrieved document.

        Args:
            idx (int): Index of the result/document in doc_info.
            score (float): Similarity (& Diversity) score of document.

        Returns:
            dict: Dictionary containing the document text and additional information.
        """

        doc = self.doc_info.iloc[idx]
        # Create the result dictionary
        result_dict = {
            "text": doc["text"],
            "doc_id": doc["org_doc_id"],
            "score": score
        }

        return result_dict