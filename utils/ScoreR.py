import re
import string
from collections import Counter
import torch
import torch.nn.functional as F
def get_freq(dataset, topn, stop_words=None):
    keys = dataset[0].keys()
    texts = []
    for key in keys:
        for i in range(len(dataset[key])):
            texts.append(dataset[i][key])
    words = []
    for text in texts:
        raw_words = re.findall(r"[A-Za-z]+|[0-9]|[+\-*/=]|[.,!?;:\"'(){}[\]]|<<|>>|\$", text)
        for word in raw_words:
            cleaned = word
            if cleaned and cleaned not in stop_words:
                words.append(cleaned)
    counter = Counter(words)
    return dict(counter.most_common(topn))
    

def get_word_vector(word, model, tokenizer):

    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False)
    tokens = tokenizer.tokenize(word, add_special_tokens=False)
    if not tokens:
        raise ValueError
    
  
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)  
    hidden_states = outputs.hidden_states[-1] 
    word_vector = torch.mean(hidden_states[0], dim=0)  
    return word_vector



def get_domain_vector(domain_words, model, tokenizer):

    vectors = []
    for word in domain_words:
        try:
            vec = get_word_vector(word, model, tokenizer)
            vectors.append(vec)
        except ValueError as e:
            print(f"Warning: {e}, has been skipped")
    if not vectors:
        raise ValueError
    return torch.mean(torch.stack(vectors), dim=0)

def normalize(values, target_min, target_max, reverse=False):
    
    v_min, v_max = min(values), max(values)
    if v_max == v_min:  
        return [1.0 for _ in values]
    if reverse:  
        return [target_max - (v - v_min) / (v_max - v_min) * (target_max - target_min) for v in values]
    else:        
        return [target_min + (v - v_min) / (v_max - v_min) * (target_max - target_min) for v in values]
def compute_relevance(vocab, domain_words, model, tokenizer, device = "cuda:0"):
    
    domain_vector = get_domain_vector(domain_words, model, tokenizer)
    

    data = {}
    distances, freqs = [], []
    for word, freq in vocab.items():
        try:
            word_vec = get_word_vector(word, model, tokenizer)
            cos_sim = F.cosine_similarity(word_vec.unsqueeze(0), domain_vector.unsqueeze(0), dim=1)
            distance = 1 - cos_sim.item()  
            data[word] = {"freq": freq, "distance": distance}
            distances.append(distance)
            freqs.append(freq)
        except ValueError as e:
            print(f"跳过词 '{word}': {e}")
    

    freq_scores = normalize(freqs, target_min=0.8, target_max=1.0)
    dist_scores = normalize(distances, target_min=0.8, target_max=1.0, reverse=True)
    
    
    results = {}
    for i, (word, info) in enumerate(data.items()):
        final_score = freq_scores[i] * dist_scores[i]
        results[word] = {
            "freq": info["freq"],
            "distance": info["distance"],
            "freq_score": freq_scores[i],
            "semantic_score": dist_scores[i],
            "final_score": final_score,
            "cos_sim": 1 - info["distance"]
        }
    return results


