import requests
from typing import List, Dict

class LocalLLMEngine:
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8080",
        *,
        default_temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.api_url = api_url
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens
        self._test_connection()
        self.default_model = self._select_default_model()

    def _test_connection(self) -> None:
        try:
            response = requests.get(f"{self.api_url}/v1/models")
            if response.status_code != 200:
                raise ConnectionError(f"Could not connect to LLM server at {self.api_url}")
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to LLM server: {str(e)}")

    def _get_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.api_url}/v1/models", timeout=10)
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = []
            # handle a few common response shapes
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                for m in data["data"]:
                    if isinstance(m, dict):
                        models.append(m.get("id") or m.get("name") or m.get("model"))
                    elif isinstance(m, str):
                        models.append(m)
            elif isinstance(data, list):
                for m in data:
                    if isinstance(m, dict):
                        models.append(m.get("id") or m.get("name") or m.get("model"))
                    elif isinstance(m, str):
                        models.append(m)
            # filter out None
            return [m for m in models if m]
        except Exception:
            return []

    def _select_default_model(self) -> str:
        models = self._get_models()
        return models[0] if models else None

    def _extract_content(self, resp_json: Dict) -> str:
        # common OpenAI-like shape
        try:
            return resp_json["choices"][0]["message"]["content"]
        except Exception:
            pass
        try:
            return resp_json["choices"][0]["text"]
        except Exception:
            pass
        try:
            return resp_json["output"][0].get("content") or resp_json["output"][0].get("text")
        except Exception:
            pass
        # fallback: return full json as string
        return str(resp_json)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        model: str = None,
    ) -> str:
        model_to_use = model or self.default_model
        if not model_to_use:
            raise Exception("No model specified and unable to auto-detect a model from the server.")
        request_temperature = (
            self.default_temperature if temperature is None else temperature
        )

        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": request_temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            
            if response.status_code == 200:
                return self._extract_content(response.json())
            raise Exception(f"API call failed ({response.status_code}): {response.text}")
        except requests.RequestException as exc:
            raise Exception(f"Request failed: {str(exc)}") from exc
