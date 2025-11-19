import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    response_time: float
    metadata: Dict[str, Any] = None


class LLMManager:
    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-pro",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        
        if provider == "gemini":
            if not gemini_api_key:
                raise ValueError("Gemini API key is required for Gemini provider")
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model_name = model.replace("models/", "") if model.startswith("models/") else model
            self.client = genai.GenerativeModel(model_name)
        elif provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")
            import openai
            self.client = openai.OpenAI(api_key=openai_api_key)
        elif provider == "anthropic":
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required for Anthropic provider")
            import anthropic
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        elif provider == "mistral":
            if not mistral_api_key:
                raise ValueError("Mistral API key is required for Mistral provider")

            from mistralai import Mistral
            self.client = Mistral(api_key=mistral_api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized LLM Manager with {provider} provider, model: {model}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> LLMResponse:
        start_time = time.time()
        
        try:
            if self.provider == "gemini":
                response = self._generate_gemini(
                    prompt, system_prompt, temperature, max_tokens
                )
            elif self.provider == "openai":
                response = self._generate_openai(
                    prompt, system_prompt, temperature, max_tokens, json_mode
                )
            elif self.provider == "anthropic":
                response = self._generate_anthropic(
                    prompt, system_prompt, temperature, max_tokens
                )
            elif self.provider == "mistral":
                response = self._generate_mistral(
                    prompt, system_prompt, temperature, max_tokens
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            response.response_time = time.time() - start_time
            logger.info(
                f"Generated response in {response.response_time:.2f}s, "
                f"tokens: {response.tokens_used}"
            )
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        json_mode: bool
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if json_mode and "gpt-4" in self.model:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            response_time=0
        )
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text if response.content else "",
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            response_time=0
        )
    
    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.95,
            "top_k": 40
        }
        
        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        content = ""
        if response.candidates:
            content = response.candidates[0].content.parts[0].text
        
        estimated_tokens = len(full_prompt) // 4 + len(content) // 4
        
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=estimated_tokens,
            response_time=0
        )

    def _generate_mistral(
            self,
            prompt: str,
            system_prompt: Optional[str],
            temperature: float,
            max_tokens: int
    ) -> LLMResponse:

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        content = ""
        if response.choices:
            content = response.choices[0].message.content

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        estimated_tokens = len(full_prompt) // 4 + len(content) // 4

        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=estimated_tokens,
            response_time=0
        )

    def generate_sql(
        self,
        question: str,
        schema_context: str,
        examples: Optional[List[Dict[str, str]]] = None,
        use_schema_reasoning: bool = True
    ) -> str:
        """
        Generate SQL with schema-guided reasoning and RAG context
        
        Args:
            question: Natural language question
            schema_context: Schema context from RAG system
            examples: Similar query examples from RAG
            use_schema_reasoning: Enable schema-guided reasoning chain-of-thought
        """
        system_prompt = """You are an expert SQL developer with deep understanding of database schemas and query optimization.

Your task is to generate accurate PostgreSQL queries using Schema-Guided Reasoning:
1. Analyze the question to identify required entities, relationships, and operations
2. Map question entities to schema tables and columns
3. Identify necessary JOINs based on foreign key relationships
4. Construct the SQL query following PostgreSQL syntax
5. Validate that all referenced tables and columns exist in the schema

CRITICAL RULES:
- Use ONLY tables and columns explicitly mentioned in the schema
- DO NOT invent or guess table/column names
- If a table/column is not in the schema, DO NOT use it
- Follow foreign key relationships for JOINs
- Use appropriate aggregation functions (COUNT, SUM, AVG, etc.)
- IMPORTANT: For SUM(), AVG(), MAX(), MIN() - use only on numeric columns. If column type is text/varchar, use CAST(column AS numeric) first
- ALWAYS add AS alias for calculated columns (e.g., AVG(column) AS avg_column, SUM(x)/COUNT(y) AS rate)
- Always add LIMIT clause for safety (max 100 rows)
- CRITICAL: For "approved loans" or "одобренные займы" - use WHERE TARGET=0 (TARGET=0 means approved)
- CRITICAL: For "defaults" or "дефолты" - use WHERE TARGET=1 (TARGET=1 means defaulted)
- CRITICAL: For "sums of loans" or "суммы займов" - use SUM(AMT_CREDIT), NOT COUNT()
- CRITICAL: For "loan amounts" - use AMT_CREDIT column, NOT loan_count
- Return ONLY the SQL query, no explanations or markdown"""

        if not schema_context or schema_context.strip() == "":
            schema_context = "No detailed schema provided. Use common PostgreSQL table naming conventions."

        if use_schema_reasoning:
            prompt = f"""Database Schema:
{schema_context}

Question: {question}

Generate SQL using Schema-Guided Reasoning:

Step 1 - Understand the Question:
- What entities are mentioned? (e.g., customers, orders, products)
- What operations are needed? (e.g., count, average, filter, group)
- What relationships are implied? (e.g., customer has orders)

Step 2 - Map to Schema:
- Which tables contain the needed entities?
- Which columns match the question requirements?
- What JOINs are needed based on foreign keys?
- CRITICAL: For "approved loans" or "одобренные займы" - use WHERE TARGET=0 (TARGET=0 means approved, TARGET=1 means defaulted)
- CRITICAL: For "defaults" or "дефолты" - use WHERE TARGET=1
- CRITICAL: For "sums of loans" or "суммы займов" - use SUM(AMT_CREDIT), NOT COUNT()
- CRITICAL: For "loan amounts" - use AMT_CREDIT column, NOT loan_count

Step 3 - Construct SQL:
- Write the SELECT clause with appropriate columns
- IMPORTANT: Add AS alias for ALL calculated columns (e.g., AVG(x) AS avg_x, SUM(x)/COUNT(y) AS rate)
- CRITICAL: For numeric aggregations (SUM, AVG, MAX, MIN) on text columns, use CAST(column AS numeric) or CAST(column AS double precision)
- Add necessary JOINs
- Apply filters (WHERE) if needed
- Add aggregations (GROUP BY, HAVING) if needed
- Order results if relevant (use CAST for numeric sorting on text columns)
- Add LIMIT for safety

Generate the SQL query:"""
        else:
            prompt = f"""Database Schema:
{schema_context}

Question: {question}

Generate a SQL query to answer this question. Use ONLY the table names mentioned in the schema above."""

        if examples:
            prompt += "\n\nSimilar Query Examples (from RAG system):\n"
            for i, ex in enumerate(examples[:3], 1):
                prompt += f"\nExample {i}:\n"
                prompt += f"Question: {ex.get('question', ex.get('natural_language_query', ''))}\n"
                prompt += f"SQL: {ex.get('sql', ex.get('sql_query', ''))}\n"
            prompt += "\nUse these examples as reference for query patterns and structure.\n"
        
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=800
        )

        sql = response.content.strip()

        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        lines = sql.split('\n')
        sql_lines = []
        in_sql = False
        for line in lines:
            if any(keyword in line.upper() for keyword in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
                in_sql = True
            if in_sql:
                sql_lines.append(line)
        
        if sql_lines:
            sql = '\n'.join(sql_lines)
        
        return sql.strip()
    
    def interpret_results(
        self,
        question: str,
        sql_query: str,
        results: str,
        domain: Optional[str] = "home credit",
        context: Optional[str] = None
    ) -> str:
        system_prompt = f"""You are a concise, {domain} business-focused data analyst. Interpret SQL query results in clear, 
        concise natural language that answers the user's question. Use ONLY the provided data/statistics/patterns to explain results.
        Do NOT invent patterns or values that are not present. Output must be short and non-repetitive.
        Use markdown formatting for better readability."""
        
        prompt = f"""User Question: {question}

SQL Query Executed:
```sql
{sql_query}
```

Results:
{results}

{f'Additional Context: {context}' if context else ''}

Provide a clear interpretation of results that answers the user's question.
Do NOT repeat the same idea multiple times.
Include numeric details only if they materially change a business decision.
Avoid statistical jargon (no p-values, coefficients, correlations) unless critical.
If relevant, add 3 or less short next steps under "### Recommended actions".
Keep the entire response concise and directly focused on what the business should know or do.
"""

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=600
        )
        
        return response.content
    
    def generate_visualization_spec(
        self,
        data_description: str,
        user_request: Optional[str] = None
    ) -> Dict[str, Any]:
        system_prompt = """You are a data visualization expert. Generate JSON specifications 
        for creating appropriate charts using Plotly. Consider the data type and user needs."""
        
        prompt = f"""Data Description:
{data_description}

{f'User Request: {user_request}' if user_request else 'Suggest the best visualization for this data.'}

Generate a JSON specification with:
- chart_type: (bar, line, scatter, pie, heatmap, box, histogram)
- title: Chart title
- x_axis: Column for x-axis (if applicable)
- y_axis: Column for y-axis (if applicable)
- color: Column for color encoding (optional)
- additional_params: Any additional parameters"""
        
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=500,
            json_mode=True
        )
        
        try:
            content = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    content = content[start:end].strip()
            
            spec = json.loads(content)
            return spec
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse visualization spec: {response.content}")
            logger.debug(f"JSON decode error: {e}")
            # Try to extract JSON object manually
            try:
                # Find first { and last }
                start = response.content.find('{')
                end = response.content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = response.content[start:end+1]
                    spec = json.loads(json_str)
                    return spec
            except:
                pass
            
            return {
                "chart_type": "bar",
                "title": "Data Visualization",
                "error": "Failed to generate specific visualization"
            }

    def translate_text(self, text: str, target_language: str = "English") -> str:
        system_prompt = (
            "You are a professional translator. Translate the user's message into "
            f"{target_language}. Return only the translated text without commentary."
        )

        try:
            response = self.generate(
                prompt=text,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=400,
            )
            return response.content.strip()
        except Exception as exc:
            logger.error("Translation failed: %s", exc)
            return text
