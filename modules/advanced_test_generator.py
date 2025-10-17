"""
Advanced AI-powered test case generation
"""
import os
from typing import Dict, List, Any
from openai import OpenAI
import json

class AdvancedTestGenerator:
    """Generate comprehensive test cases using LLM"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o" if os.getenv("TG_USE_GPT4") == "true" else "gpt-4o-mini"
    
    def generate_edge_cases(self, feature: str) -> List[Dict[str, Any]]:
        """Generate edge case tests for a feature"""
        
        prompt = f"""
Generate 10 edge case test scenarios for: {feature}

Include:
- Boundary conditions
- Invalid inputs
- Race conditions
- Error scenarios
- Security tests

Return as JSON:
{{
  "test_cases": [
    {{
      "name": "Test name",
      "steps": ["step1", "step2"],
      "expected": "expected result",
      "priority": "P0/P1/P2",
      "risk_score": 80
    }}
  ]
}}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("test_cases", [])
