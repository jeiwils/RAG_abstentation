
import torch
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from .x_utils import load_model_8bit, parse_llm_json
from typing import List, Dict






class NLIModel:
    def __init__(self, model_name="roberta-large-mnli", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, answer: str, passage: str, reward_scheme: dict):
        inputs = self.tokenizer(passage, answer, return_tensors="pt", truncation=True).to(self.device)
        logits = self.model(**inputs).logits
        pred_class = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
        labels = ["contradiction", "neutral", "entailment"]
        label = labels[pred_class]
        nli_rewards = reward_scheme.get("nli", {"entailment": 0.5, "neutral": -0.2, "contradiction": -1.0})
        return {
            "entailment": nli_rewards.get("entailment", 0.5) if label == "entailment" else 0.0,
            "neutral": nli_rewards.get("neutral", -0.2) if label == "neutral" else 0.0,
            "contradiction": nli_rewards.get("contradiction", -1.0) if label == "contradiction" else 0.0,
        }

    def free(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()




class EmbedModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], normalize=True):
        return self.model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)

    def free(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()






class LLMModel:
    def __init__(self, model_path, fp32_offload=False):
        self.tokenizer, self.model = load_model_8bit(model_path, enable_fp32_cpu_offload=fp32_offload)

    def generate(self, prompt: str, gen_params: dict):
        """
        Generates an answer from the model using a fully prepared prompt.
        """
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        outputs = self.model.generate(**inputs, **gen_params)
        sequence = outputs.sequences[0]
        raw_text = self.tokenizer.decode(sequence, skip_special_tokens=True)

        # Parse JSON from model output
        parsed = parse_llm_json(raw_text)

        # Wrap in dict for pipeline
        cand = {
            "answer": parsed.get("answer", "").strip(),
            "self_conf": float(parsed.get("confidence", 0.0)),
            "idk": "i don't know" in parsed.get("answer", "").lower() or parsed.get("answer", "").lower().startswith("idk"),
            "cited_passages": parsed.get("cited_passages", []),
            "raw_output": raw_text,
            "cleaned_output": parsed.get("cleaned_output", ""),
            "error": parsed.get("error")
        }
        return cand

    def free(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
















# def generate_answer(
#         model, 
#         tokenizer, 
#         query, 
#         passages, 
#         gen_params, 
#         prompt_path=ANSWER_PROMPT
#         ):
#     prompt = fill_prompt(prompt_path, query, passages)
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

#     outputs = model.generate(
#         **inputs,
#         **gen_params,
#     )
#     sequence = outputs.sequences[0]
#     raw_text = tokenizer.decode(sequence, skip_special_tokens=True)

#     parsed = parse_llm_json(raw_text)

#     answer_text = parsed.get("answer", "").strip()
#     self_conf = float(parsed.get("confidence", 0.0))
#     cited_idxs = parsed.get("cited_passages", [])
#     is_idk = "i don't know" in answer_text.lower() or answer_text.lower().startswith("idk")

#     return {
#         "answer": answer_text,
#         "self_conf": self_conf,
#         "idk": is_idk,
#         "cited_passages": cited_idxs,
#         "raw_output": parsed.get("raw_output", raw_text),
#         "cleaned_output": parsed.get("cleaned_output", ""),
#         "error": parsed.get("error")
#     }













# def score_NLI_fn(
#         answer, 
#         passage, 
#         reward_scheme
#         ):
#     """

    
#     """
#     inputs = tokenizer_nli(passage, answer, return_tensors="pt", truncation=True).to("cuda")
#     with torch.no_grad():
#         logits = model_nli(**inputs).logits
#         pred_class = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()

#     mnli_labels = ["contradiction", "neutral", "entailment"]
#     label = mnli_labels[pred_class]

#     nli_rewards = reward_scheme.get("nli", {"entailment": 0.5, "neutral": -0.2, "contradiction": -1.0})
#     return {
#         "entailment": nli_rewards.get("entailment", 0.5) if label == "entailment" else 0.0,
#         "neutral": nli_rewards.get("neutral", -0.2) if label == "neutral" else 0.0,
#         "contradiction": nli_rewards.get("contradiction", -1.0) if label == "contradiction" else 0.0,
#     }






# def NLI_check_answer_per_passage(
#         answer, 
#         passages, 
#         reward_scheme
#         ):
#     """

#     """
#     nli_rewards = reward_scheme.get("nli", {"entailment": 0.5, "neutral": -0.2, "contradiction": -1.0})
#     scores = []

#     for passage in passages:
#         inputs = tokenizer_nli(passage, answer, return_tensors="pt", truncation=True).to("cuda")
#         with torch.no_grad():
#             logits = model_nli(**inputs).logits
#             pred_class = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()

#         mnli_labels = ["contradiction", "neutral", "entailment"]
#         label = mnli_labels[pred_class]
#         scores.append(nli_rewards[label])

#     return scores
