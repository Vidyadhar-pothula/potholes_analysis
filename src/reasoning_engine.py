import os
import requests
import json

class ReasoningEngine:
    def __init__(self, ollama_url="http://localhost:11434", model_name="llama3"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.use_hf_fallback = False
        
        # Check if ollama is running
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if resp.status_code == 200:
                print(f"Connected to Ollama. Using {model_name}.")
            else:
                self.use_hf_fallback = True
        except Exception:
            print("Ollama not reachable, falling back to HuggingFace pipeline (if installed)...")
            self.use_hf_fallback = True

        if self.use_hf_fallback:
            try:
                from transformers import pipeline
                print("Loading lightweight HuggingFace model (flan-t5-small)...")
                # Using a very lightweight model for quick inference
                self.hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
            except ImportError:
                print("transformers library not found. Will use a rule-based dummy responder.")
                self.hf_pipeline = None

    def construct_context(self, stats_dict):
        num_potholes = stats_dict.get('num_potholes', 0)
        features = stats_dict.get('features', [])
        
        if num_potholes == 0:
            return "There are no potholes detected on this road segment."
            
        context = f"There are {num_potholes} potholes detected.\n"
        for idx, feat in enumerate(features):
            context += f"Pothole {idx+1} (ID: {feat['pothole_id']}) has area {feat.get('area_m2', 0)} m2, depth category {feat.get('depth_cat', 'Unknown')}, max depth {feat.get('max_depth', 0):.2f}, and severity is {feat.get('severity', 'Unknown')}.\n"

        
        context += "Based on the severity (High means urgent repair needed, Medium means monitor, Low means safe for now)."
        return context

    def get_response(self, question, stats_dict):
        context = self.construct_context(stats_dict)
        
        prompt = f"""Context information is below:
---------------------
{context}
---------------------
Given the context information, answer the following user question:
Question: {question}
Answer:"""

        if not self.use_hf_fallback:
            # Try Ollama API
            try:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
                response = requests.post(f"{self.ollama_url}/api/generate", json=data, timeout=30)
                if response.status_code == 200:
                    return response.json().get("response", "No response generated.")
            except Exception as e:
                print(f"Ollama generation failed: {e}")
                pass # fall through to fallback if possible

        # HF Fallback or Dummy
        if self.use_hf_fallback and self.hf_pipeline:
            # FLAN-T5 expects task instructions
            input_text = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}"
            try:
                res = self.hf_pipeline(input_text, max_length=100)
                return res[0]['generated_text']
            except Exception as e:
                return f"Error with HF model: {e}"
        else:
            # Rule based dummy fallback if no models available
            question = question.lower()
            if "how many" in question:
                return f"There are {stats_dict.get('num_potholes', 0)} potholes detected."
            elif "safe" in question:
                high_sev = any(f['severity'] == 'High' for f in stats_dict.get('features', []))
                if high_sev:
                    return "This road is not safe. There are high severity potholes."
                return "The road is relatively safe, but monitor the detected potholes."
            elif "urgent" in question or "repair" in question:
                high_sev = sum(1 for f in stats_dict.get('features', []) if f['severity'] == 'High')
                if high_sev > 0:
                    return f"Yes, {high_sev} potholes require urgent repair."
                return "No urgent repairs needed at this time."
            else:
                return "I'm a simple assistant. Based on the data, there are " + str(stats_dict.get('num_potholes', 0)) + " potholes."
