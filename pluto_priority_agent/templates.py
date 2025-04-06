temp = \
"""
You are a domain expert in healthcare innovation and risk assessment.

Your task is to evaluate the risk and potential of a healthcare technology idea based on its dataset entry and broader industry knowledge.  
Use step-by-step reasoning (ReAct-style) and simulate external research into comparable technologies, regulations, and market trends.

---

### [INPUT: IDEA DETAILS]

ID: {id}  
Title: {title}  
Description: {description}  

---

### Research each metric and generate a score. Each score must be between 0.0 and 10.0.
### Please return your entire response strictly as a valid JSON object (no markdown, no extra text). 
### Ensure that all keys in the JSON below are present
### Only return the JSON object below. Do not include any markdown formatting or additional text.


{{
  "s1": {{
    "impact_metrics": {{
      "value_created": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }},
      "user_demand": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }},
      "business_impact": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }}
    }},
    "effort_metrics": {{
      "time": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }},
      "resources": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }},
      "dependencies": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }}
    }},
    "other_metrics": {{
      "feasibility_concerns": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }},
      "risk": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }},
      "uncertainties_or_gaps": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }},
      "similar_tech_performance": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }},
      "user_adoption": {{
        "score": <float>,
        "explanation": "<short explanation>"
      }}
    }}
  }},
  "final_output": {{
    "thought": "<Provide a concise explanation for the scores assigned in each metric category in 100 words>"
  }}
}}
"""

temp2 = \
"""
I’ve used a custom scoring function to rank and select the top three project ideas from a larger set. The function considers three dimensions:
The user selects weights for these dimensions to prioritize one over the other.

1. Impact (average of value created, user demand, business impact)
2. Effort (inverse average of time, resources, and dependencies)
3. Other Factors (inverse of feasibility concerns, risk, uncertainties, plus similar tech score)

Each score is normalized (divided by 10), and a final weighted sum is calculated:

final_score = impact_weight * impact_score + effort_weight * effort_score + other_weight * other_score

I’ve used the following weights:
- impact_weight = {impact_weight}
- effort_weight = {effort_weight}
- other_weight = {other_weight}

This is the json data from the dataframe of the top three ideas {json_data}


Only return the below reasoning in 150 words.
1. Why these three ideas were selected, based on the impact, effort, and other metrics. Reason in simple layman language. donot focus on rank score but 
2. How the choice of weights (impact_weight, effort_weight, other_weight) for user influenced this selection.
"""



