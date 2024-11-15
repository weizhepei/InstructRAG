## Datasets
### Question-Answering (QA) Pairs
The raw QA datasets used in our work can be found in this HF dataset repo: [meng-lab/InstructRAG](https://huggingface.co/datasets/meng-lab/InstructRAG).

### Retrieval-augmented Dataset
You can also directly download the pre-processed datasets (augmented with retrieved documents and rationales) from this [link](https://drive.google.com/file/d/1MVkdc4g9_D4REtaBFKeJ9gMun4qzdQtO/view?usp=share_link).


Please refer to the [rationale generation script](../generate_rationale.sh) for detailed instructions on preparing data with your own corpus.

## Retrieval with customized queries
As stated above, we have provided retrieved documents along with the queries for all datasets used in this work to facilitate easier reproduction. To perform retrieval with customized queries, the easiest way is to use [Pyserini](https://github.com/castorini/pyserini) with prebuilt indexes of retrieval corpus (e.g., Wikipedia). Below are some code snippets for sparse retrieval (e.g., BM25) and dense retrieval (e.g., DPR) for your reference.

- Sparse Retrieval
```python
# Sparse Retriever (BM25)
from pyserini.search.lucene import LuceneSearcher

# Use Wikipedia dump as the retrieval source
searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr') 
# Retrieve documents relevant to the given query
hits = searcher.search('who got the first nobel prize in physics')
# Present retrieved document and relevance score
print(f'doc: {searcher.doc(hits[0].docid).raw()}\nscore: {hits[0].score}')
```

- Dense Retrieval

```python
# Dense Retriever (DPR)
from pyserini.search.faiss import FaissSearcher, DprQueryEncoder

# Load query encoder
encoder = DprQueryEncoder("facebook/dpr-question_encoder-single-nq-base")
# Use Wikipedia dump as the retrieval source
searcher = FaissSearcher.from_prebuilt_index('wikipedia-dpr-100w.dpr-single-nq', encoder)
# Retrieve documents relevant to the given query
hits = searcher.search('who got the first nobel prize in physics')
# Present retrieved document and relevance score
print(f'doc: {searcher.doc(hits[0].docid).raw()}\nscore: {hits[0].score}')
```
