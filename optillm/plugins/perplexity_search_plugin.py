import os
import re
import requests
from typing import Tuple, List, Dict, Optional

SLUG = "perplexity_search"

# Perplexity API endpoint
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def extract_search_queries(text: str) -> List[str]:
    """Extract potential search queries from the input text"""
    # Clean up common prefixes from chat messages
    text = text.strip()
    # Remove common role prefixes
    for prefix in ["User:", "user:", "User ", "user ", "Assistant:", "assistant:", "System:", "system:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Look for explicit search requests
    # Note: Removed period (.) from exclusion to allow queries like "Python 3.12" to work
    # Updated to require at least one non-whitespace character after the prefix
    search_patterns = [
        r"search for[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
        r"find information about[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
        r"look up[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
        r"research[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
    ]

    queries = []
    for pattern in search_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            cleaned = match.strip()
            # Remove trailing quotes (single or double)
            cleaned = cleaned.rstrip('"\'')
            # Remove leading quotes if they exist
            cleaned = cleaned.lstrip('"\'')
            # Only add non-empty queries
            if cleaned:
                queries.append(cleaned)

    # If no explicit patterns, use the text as a search query
    # Since the user explicitly invoked the perplexity_search plugin, we should always search
    if not queries:
        # Check if this is a search command with empty query (e.g., "search for" with nothing after)
        search_prefixes = ["search for", "find information about", "look up", "research"]
        text_lower = text.lower().strip()

        # Don't use fallback if it's just a search prefix with nothing meaningful after
        is_empty_search = any(
            text_lower.startswith(prefix) and
            len(text_lower.replace(prefix, "").strip().strip('"\'')) < 2
            for prefix in search_prefixes
        )

        if not is_empty_search and text.strip():
            # User explicitly invoked perplexity_search plugin - send the full query
            cleaned_query = text.strip()

            # Just basic cleanup: normalize whitespace (replace newlines with spaces)
            cleaned_query = ' '.join(cleaned_query.split())

            # Final validation - must have some meaningful content
            if cleaned_query and len(cleaned_query) >= 3:
                queries.append(cleaned_query)

    return queries


def perplexity_search(
    query: str,
    api_key: str,
    model: str = "sonar",
    num_results: int = 10,
    search_mode: str = "web",
    search_recency_filter: Optional[str] = None,
    search_domain_filter: Optional[List[str]] = None,
    return_related_questions: bool = False,
) -> Dict:
    """
    Perform a search using the Perplexity API.

    Args:
        query: The search query
        api_key: Perplexity API key
        model: Model to use (sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro)
        num_results: Not directly supported by Perplexity, but we can request more context
        search_mode: Search mode (web, academic, sec)
        search_recency_filter: Filter by time period (day, week, month, year)
        search_domain_filter: List of domains to include/exclude
        return_related_questions: Whether to return related questions

    Returns:
        Dictionary with 'answer' (str), 'citations' (list), and 'related_questions' (list)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful research assistant. Provide accurate, well-sourced information based on web search results. Be concise but comprehensive."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "search_mode": search_mode,
        "return_related_questions": return_related_questions,
    }

    # Add optional filters
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter

    if search_domain_filter:
        payload["search_domain_filter"] = search_domain_filter

    try:
        print(f"🔍 Perplexity search: {query[:50]}...")
        response = requests.post(
            PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()

        # Extract the answer from choices
        answer = ""
        if result.get("choices") and len(result["choices"]) > 0:
            answer = result["choices"][0].get("message", {}).get("content", "")

        # Extract citations/search results if available
        citations = result.get("citations", [])

        # Also check for search_results field (alternative format)
        search_results = result.get("search_results", [])

        # Extract related questions if available
        related_questions = result.get("related_questions", [])

        # Token usage for logging
        usage = result.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        print(f"✅ Perplexity search completed (tokens: {total_tokens})")

        return {
            "answer": answer,
            "citations": citations or search_results,
            "related_questions": related_questions,
            "usage": usage
        }

    except requests.exceptions.HTTPError as e:
        error_msg = f"Perplexity API HTTP error: {e}"
        if e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = f"Perplexity API error: {error_detail}"
            except:
                error_msg = f"Perplexity API HTTP error: {e.response.status_code} - {e.response.text}"
        print(f"❌ {error_msg}")
        return {"answer": "", "citations": [], "related_questions": [], "error": error_msg}

    except requests.exceptions.Timeout:
        error_msg = "Perplexity API request timed out"
        print(f"❌ {error_msg}")
        return {"answer": "", "citations": [], "related_questions": [], "error": error_msg}

    except requests.exceptions.RequestException as e:
        error_msg = f"Perplexity API request error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"answer": "", "citations": [], "related_questions": [], "error": error_msg}

    except Exception as e:
        error_msg = f"Unexpected error during Perplexity search: {str(e)}"
        print(f"❌ {error_msg}")
        return {"answer": "", "citations": [], "related_questions": [], "error": error_msg}


def format_search_results(query: str, result: Dict) -> str:
    """Format Perplexity search results into readable text"""
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    related_questions = result.get("related_questions", [])
    error = result.get("error")

    if error:
        return f"Search error for '{query}': {error}"

    if not answer:
        return f"No search results found for: {query}"

    formatted = f"Search results for '{query}':\n\n"
    formatted += f"**Answer:**\n{answer}\n\n"

    if citations:
        formatted += "**Sources:**\n"
        for i, citation in enumerate(citations, 1):
            if isinstance(citation, dict):
                title = citation.get("title", "Source")
                url = citation.get("url", "")
                date = citation.get("date", "")
                formatted += f"{i}. [{title}]({url})"
                if date:
                    formatted += f" ({date})"
                formatted += "\n"
            elif isinstance(citation, str):
                # Citation might just be a URL string
                formatted += f"{i}. {citation}\n"
        formatted += "\n"

    if related_questions:
        formatted += "**Related Questions:**\n"
        for question in related_questions:
            formatted += f"- {question}\n"
        formatted += "\n"

    return formatted


def run(system_prompt: str, initial_query: str, client=None, model: str = None, request_config: Optional[Dict] = None) -> Tuple[str, int]:
    """
    Perplexity search plugin that uses the Perplexity API for web search.

    Args:
        system_prompt: System prompt for the conversation
        initial_query: User's query that may contain search requests
        client: OpenAI client (unused for this plugin)
        model: Model name (unused for this plugin, we use Perplexity models)
        request_config: Optional configuration dict with keys:
            - perplexity_model: Perplexity model to use (default: "sonar")
            - num_results: Number of results hint (default: 10)
            - search_mode: Search mode - web, academic, sec (default: "web")
            - search_recency_filter: Filter by time - day, week, month, year (default: None)
            - search_domain_filter: List of domains to filter (default: None)
            - return_related_questions: Include related questions (default: False)

    Returns:
        Tuple of (enhanced_query_with_search_results, completion_tokens)
    """
    # Get API key from environment
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        error_msg = "PERPLEXITY_API_KEY environment variable not set"
        print(f"❌ {error_msg}")
        enhanced_query = f"{initial_query}\n\n[Perplexity Search Error]: {error_msg}"
        return enhanced_query, 0

    # Parse configuration
    config = request_config or {}
    perplexity_model = config.get("perplexity_model", "sonar")
    num_results = config.get("num_results", 10)
    search_mode = config.get("search_mode", "web")
    search_recency_filter = config.get("search_recency_filter", None)
    search_domain_filter = config.get("search_domain_filter", None)
    return_related_questions = config.get("return_related_questions", False)

    # Extract search queries from the input
    search_queries = extract_search_queries(initial_query)

    if not search_queries:
        return initial_query, 0

    enhanced_query = initial_query
    total_tokens = 0

    for query in search_queries:
        # Perform the search
        result = perplexity_search(
            query=query,
            api_key=api_key,
            model=perplexity_model,
            num_results=num_results,
            search_mode=search_mode,
            search_recency_filter=search_recency_filter,
            search_domain_filter=search_domain_filter,
            return_related_questions=return_related_questions,
        )

        # Track token usage
        usage = result.get("usage", {})
        total_tokens += usage.get("total_tokens", 0)

        # Format results
        if result.get("answer") or result.get("citations"):
            formatted_results = format_search_results(query, result)
            enhanced_query = f"{enhanced_query}\n\n[Perplexity Search Results]:\n{formatted_results}"
        elif result.get("error"):
            enhanced_query = f"{enhanced_query}\n\n[Perplexity Search Error]: {result['error']}"
        else:
            enhanced_query = f"{enhanced_query}\n\n[Perplexity Search Results]:\nNo results found for '{query}'."

    return enhanced_query, total_tokens
