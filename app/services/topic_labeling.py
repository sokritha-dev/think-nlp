from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def auto_match_labels(
    topics: list[dict], user_keywords: list[str]
) -> tuple[dict[int, str], list[dict]]:
    label_map = {}
    detailed_results = []

    user_embeds = model.encode(user_keywords, convert_to_tensor=True)

    for topic in topics:
        topic_id = int(topic["topic_id"])
        topic_keywords = topic["keywords"].replace(",", " ")
        topic_embed = model.encode(topic_keywords, convert_to_tensor=True)

        sims = util.cos_sim(topic_embed, user_embeds)[0]  # 1 x N
        best_idx = sims.argmax().item()
        best_score = sims[best_idx].item()
        matched_label = user_keywords[best_idx]

        label_map[topic_id] = matched_label
        detailed_results.append(
            {
                "topic_id": topic["topic_id"],
                "keywords": topic["keywords"],
                "label": matched_label,
                "confidence": round(best_score, 4),
                "matched_with": "auto match keywords"
            }
        )

    return label_map, detailed_results


def generate_default_labels(
    topics: list[dict], num_keywords: int = 2
) -> dict[int, str]:
    """
    Auto-label topics using top-k keywords per topic.
    E.g., topic with keywords 'hotel, room, bed' becomes 'hotel & room'
    """
    labels = {}
    for topic in topics:
        topic_id = int(topic["topic_id"])
        keywords = topic["keywords"].split(", ")
        label_keywords = keywords[:num_keywords]
        labels[topic_id] = " & ".join(label_keywords)
    return labels
