from typing import List, Literal, Optional, Tuple
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def apply_lda_model(
    token_series: pd.Series, num_topics: int = 5
) -> Tuple[List[int], List[dict]]:
    """
    Apply LDA topic modeling and return topic assignments + topic summary.
    """
    tokenized = token_series.apply(eval)  # Convert stringified list to list of tokens
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda_model = models.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=10
    )

    # Assign dominant topic to each row
    dominant_topics = []
    for bow in corpus:
        topic_probs = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
        dominant_topics.append(dominant_topic)

    # Topic summary
    topics = [
        {
            "topic_id": str(i),
            "keywords": ", ".join([word for word, _ in lda_model.show_topic(i)]),
            "label": f"topic_{i}",
        }
        for i in range(num_topics)
    ]

    return dominant_topics, topics


def estimate_best_num_topics(
    docs: pd.Series,
    min_topics: int = 3,
    max_topics: int = 10,
    method: Optional[Literal["perplexity", "coherence"]] = "perplexity",
) -> int:
    """
    Estimate the best number of topics using either:
    - perplexity (default)
    - coherence ("c_v")
    """
    texts = docs.apply(eval).tolist()  # Convert stringified token lists

    if method == "coherence":
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        best_k = min_topics
        best_score = float("-inf")

        for k in range(min_topics, max_topics + 1):
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=10,
                random_state=42,
            )
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=dictionary,
                coherence="c_v",
            )
            coherence_score = coherence_model.get_coherence()
            if coherence_score > best_score:
                best_score = coherence_score
                best_k = k

        return best_k

    else:  # Default: perplexity
        text_strs = [" ".join(tokens) for tokens in texts]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(text_strs)

        best_k = min_topics
        best_score = float("inf")

        for k in range(min_topics, max_topics + 1):
            lda = LatentDirichletAllocation(n_components=k, random_state=42)
            lda.fit(X)
            score = lda.perplexity(X)
            if score < best_score:
                best_score = score
                best_k = k

        return best_k
