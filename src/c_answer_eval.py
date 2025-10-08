"""



Actual answer = run NLI check of answer against retrieved passages -> outputs Supported/Contradicted/Unsupported
IDK = check of abstention was appropriate (whether or not gold was in retrieved passages)






extras:
- I can also test NLI vs entailment model? But I think I'd need to tune an entailment model 












pseudocode::



results = []
for output in model_outputs:
    if output.answer != "IDK":
        grounding = entailment_model.check(output.answer, retrieved_passages)
        results.append({"answer": output.answer, "confidence": output.confidence, "grounding": grounding})
    else:
        safe = check_safe_abstention(retrieved_passages, gold_passages)
        results.append({"answer": "IDK", "confidence": output.confidence, "safe_abstention": safe})





"""