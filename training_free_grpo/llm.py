import time
import openai
from utu.utils import EnvUtils

class LLM:
    def __init__(self):
        EnvUtils.assert_env(["UTU_LLM_TYPE", "UTU_LLM_MODEL", "UTU_LLM_BASE_URL", "UTU_LLM_API_KEY"])
        self.model_name = EnvUtils.get_env("UTU_LLM_MODEL")
        self.client = openai.OpenAI(
            api_key=EnvUtils.get_env("UTU_LLM_API_KEY"),
            base_url=EnvUtils.get_env("UTU_LLM_BASE_URL"),
        )

    def chat(self, messages_or_prompt, max_tokens=16384, temperature=0, max_retries=5, return_reasoning=False):
        """
        Send a chat completion request with retry logic and exponential backoff.

        Args:
            messages_or_prompt: String prompt or list of message dicts
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts (default: 5)
            return_reasoning: Whether to return reasoning content

        Returns:
            Response text, or tuple of (response_text, reasoning) if return_reasoning=True
        """
        base_delay = 10  # Base delay in seconds

        for attempt in range(max_retries):
            try:
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                elif isinstance(messages_or_prompt, list):
                    messages = messages_or_prompt
                else:
                    raise ValueError("messages_or_prompt must be a string or a list of messages.")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response_text = response.choices[0].message.content.strip()

                if return_reasoning:
                    reasoning = response.choices[0].message.reasoning_content
                    return response_text, reasoning
                return response_text

            except openai.RateLimitError as e:
                # Handle rate limit errors with exponential backoff
                if attempt < max_retries - 1:
                    # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                    wait_time = base_delay * (2 ** attempt)
                    print(f"Rate limit reached: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Rate limit error after {max_retries} attempts: {e}")
                    raise

            except openai.APIError as e:
                # Handle API errors (e.g., server errors)
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"API error: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"API error after {max_retries} attempts: {e}")
                    raise

            except Exception as e:
                # Handle other unexpected errors
                if attempt < max_retries - 1:
                    wait_time = base_delay
                    print(f"Unexpected error: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    error = f"An unexpected error occurred after {max_retries} attempts: {e}"
                    print(error)
                    raise

        # This should not be reached, but just in case
        raise RuntimeError(f"Failed to get response after {max_retries} retries")