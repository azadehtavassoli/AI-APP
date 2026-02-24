"""
RAG Service for LLM Answer Generation

This service handles the Retrieval-Augmented Generation (RAG) pipeline:
1. Retrieve relevant chunks from vector store
2. Build context from retrieved chunks
3. Call tools if needed (new in v2.1)
4. Generate answer using OpenAI LLM
5. Build citations for sources

Key Components:
- generate_answer(): Main RAG function
- build_citations(): Extract citation info from chunks
- format_context(): Format chunks into context string
- route_tools(): Tool routing logic (v2.1)

Dependencies:
- External: OpenAI for LLM, LangChain for utilities
- Internal: vector_store for retrieval, settings for config, tools for tool calling

Author: Team
Last Modified: 2026-02-11
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from langchain_openai import ChatOpenAI

from src.core.settings import get_settings
from src.core.logging_setup import get_logger
from src.services.vector_store import similarity_search_with_score
from src.models.schemas import Citation

logger = get_logger(__name__)
settings = get_settings()


MODEL_PRICING_USD_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

SUPPORTED_OPENAI_MODELS = tuple(MODEL_PRICING_USD_PER_1M.keys())


def _extract_token_usage(response: Any) -> Dict[str, int]:
    """Extract token usage from LangChain/OpenAI response metadata.

    Args:
        response (Any): LLM response object returned by LangChain model invocation.

    Returns:
        Dict[str, int]: Normalized token usage with prompt/completion/total keys.
    """
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
        completion_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
        total_tokens = int(usage_metadata.get("total_tokens", 0) or 0)

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage", {})
        if isinstance(token_usage, dict):
            prompt_tokens = int(token_usage.get("prompt_tokens", prompt_tokens) or prompt_tokens)
            completion_tokens = int(token_usage.get("completion_tokens", completion_tokens) or completion_tokens)
            total_tokens = int(token_usage.get("total_tokens", total_tokens) or total_tokens)

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Estimate request cost in USD based on per-model token pricing.

    Args:
        model_name (str): OpenAI model name used for generation.
        prompt_tokens (int): Number of prompt/input tokens consumed.
        completion_tokens (int): Number of completion/output tokens consumed.

    Returns:
        Optional[float]: Estimated USD cost, or None when model pricing is unavailable.
    """
    if not model_name:
        return None

    pricing = MODEL_PRICING_USD_PER_1M.get(model_name)
    if not pricing:
        return None

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)


def _resolve_generation_model(model_override: Optional[str]) -> str:
    """Resolve final OpenAI generation model from request override and settings.

    Args:
        model_override (Optional[str]): Optional model requested by frontend for this query.

    Returns:
        str: Valid model name used for generation.
    """
    if model_override and model_override in SUPPORTED_OPENAI_MODELS:
        return model_override

    return settings.openai_model


def is_deadline_intent(text: str) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Detect Deadline/Date Arithmetic Intent
    
    Uses keyword + regex matching to detect if user is asking
    for deadline calculation (date + days arithmetic).
    
    This is a simple heuristic that triggers tool-first routing,
    bypassing retrieval for pure deadline arithmetic questions.
    
    Detection logic:
    1. Check for deadline keywords (multilingual)
    2. Check for date pattern (ISO, German, European slash)
    3. Check for number of days pattern
    4. If all present, extract parameters
    
    Args:
        text: User's question
    
    Returns:
        Tuple of (is_deadline_question, extracted_params)
        extracted_params: Dict with start_date, days, business_days if detected
    
    Example:
        >>> is_deadline_intent("Start date 01.03.2026, add 14 days")
        (True, {"start_date": "01.03.2026", "days": 14, "business_days": False})
        
        >>> is_deadline_intent("What documents do I need for visa?")
        (False, None)
    """
    if not text or not text.strip():
        return False, None
    
    text_lower = text.lower()
    
    # Keywords that indicate deadline/date arithmetic intent
    # Multilingual: English, German, Persian
    deadline_keywords = [
        # English
        'deadline', 'due date', 'add days', 'plus days', 'after days',
        'within', 'calculate date', 'date arithmetic', 'days from',
        'days after', 'start date',
        # German
        'frist', 'termin', 'tage', 'innerhalb', 'ab', 'nach', 'bis wann',
        'startdatum', 'enddatum', 'arbeitstage', 'werktage',
        # Persian (transliterated keywords for detection)
        'مهلت', 'تاریخ', 'روز', 'چند روز'
    ]
    
    # Check if any keyword present (word-aware to avoid false positives)
    has_keyword = any(
        re.search(rf"\b{re.escape(keyword)}\b", text_lower) for keyword in deadline_keywords
    )
    
    if not has_keyword:
        return False, None
    
    # Date patterns (ISO, German, European slash)
    date_patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',  # ISO: 2026-03-01
        r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b',  # German: 01.03.2026
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # Slash: 01/03/2026
        r'\b(\d{4}-\d{2}-\d{2}T[\d:]+)\b'  # ISO datetime: 2026-03-01T10:30:00
    ]
    
    # Days pattern (number + "days" or "tage")
    days_patterns = [
        r'\b(\d+)\s*(?:day|days)\b',
        r'\b(\d+)\s*(?:tag|tage)\b',
        r'\b(\d+)\s*(?:روز)\b',  # Persian "days"
        r'add\s+(\d+)',
        r'plus\s+(\d+)',
        r'after\s+(\d+)',
        r'innerhalb\s+(\d+)',
        r'within\s+(\d+)'
    ]
    
    # Extract date
    start_date = None
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_date = match.group(1)
            break
    
    # Extract days
    days = None
    for pattern in days_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                days = int(match.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # Check for business days keywords
    business_days_keywords = [
        'business day', 'work day', 'working day', 'weekday',
        'arbeitstag', 'werktag', 'geschäftstag'
    ]
    business_days = any(keyword in text_lower for keyword in business_days_keywords)
    
    # If we found both date and days, this is a deadline intent
    if start_date and days is not None:
        logger.info(f"Deadline intent detected: start_date={start_date}, days={days}, business={business_days}")
        return True, {
            "start_date": start_date,
            "days": days,
            "business_days": business_days
        }
    
    # If we found keywords but not both parameters, still might be deadline intent
    # but we need to extract from context later
    if has_keyword and (start_date or days is not None):
        logger.info(f"Partial deadline intent: start_date={start_date}, days={days}")
        return True, {
            "start_date": start_date,
            "days": days,
            "business_days": business_days
        }
    
    return False, None


def detect_language(text: str) -> str:
    """
    Detect Language of Text
    
    Uses simple heuristics to detect language:
    - Persian: Contains Persian/Arabic script characters
    - German: Contains umlauts or common German stopwords
    - English: Default fallback
    
    Args:
        text: Input text to analyze
    
    Returns:
        str: Language code ("de", "en", or "fa")
    
    Example:
        >>> detect_language("Was ist die Frist?")
        "de"
        >>> detect_language("What is the deadline?")
        "en"
    """
    if not text:
        return "en"
    
    # Check for Persian/Arabic script (U+0600 to U+06FF)
    if any('\u0600' <= c <= '\u06FF' for c in text):
        logger.debug(f"Detected Persian text")
        return "fa"
    
    # Check for German umlauts
    text_lower = text.lower()
    if any(c in text_lower for c in ['ä', 'ö', 'ü', 'ß']):
        logger.debug(f"Detected German text (umlauts)")
        return "de"
    
    # Check for common German stopwords
    german_stopwords = ['der', 'die', 'das', 'und', 'ist', 'mit', 'für', 'von', 'auf', 'dem']
    words = text_lower.split()
    german_word_count = sum(1 for word in words if word in german_stopwords)
    
    if german_word_count >= 2:  # At least 2 German words
        logger.debug(f"Detected German text (stopwords)")
        return "de"
    
    # Default to English
    logger.debug(f"Detected English text (default)")
    return "en"


def translate_query(text: str, target_lang: str) -> Optional[str]:
    """
    Translate Query to Target Language
    
    Uses OpenAI LLM to translate query for retrieval purposes.
    This is a best-effort translation - if it fails, returns None.
    
    Args:
        text: Query text to translate
        target_lang: Target language code ("de", "en", or "fa")
    
    Returns:
        Optional[str]: Translated text, or None if translation fails
    
    Example:
        >>> translate_query("What is the deadline?", "de")
        "Was ist die Frist?"
    """
    if not text or not text.strip():
        return None
    
    lang_names = {"de": "German", "en": "English", "fa": "Persian"}
    target_name = lang_names.get(target_lang, target_lang)
    
    try:
        logger.debug(
            "translate_query_call: model=%s target=%s input_chars=%s",
            settings.openai_model,
            target_lang,
            len(text),
        )
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,  # Deterministic for translations
            openai_api_key=settings.openai_api_key
        )
        
        prompt = f"""Translate the following question to {target_name}. 
Provide ONLY the translation, no explanations.

Question: {text}

Translation:"""
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        translation = response.content.strip()
        
        logger.debug(f"Translated '{text[:50]}...' to {target_lang}: '{translation[:50]}...'")
        return translation
    
    except Exception as e:
        logger.warning(f"Translation to {target_lang} failed: {e}")
        return None


def multi_pass_retrieval(
    question: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-pass Retrieval with Query Translation
    
    Retrieves documents using multiple query variants in different languages
    to handle mixed-language knowledge bases. Improves recall for cross-lingual
    document retrieval.
    
    Process:
    1. Detect question language
    2. Build query variants (original + translation)
    3. Retrieve with each variant (k=3 per variant)
    4. Merge and deduplicate results
    5. Return top_k results
    
    Args:
        question: User's question in any language
        top_k: Final number of results to return (default: 5)
        where: Optional metadata filters passed to vector search
    
    Returns:
        List[Dict]: Deduplicated and ranked list of retrieved chunks
    
    Example:
        Question: "What is the deadline?" (EN)
        Variants: ["What is the deadline?", "Was ist die Frist?"]
        Searches both variants, merges results
    """
    # Detect source language
    source_lang = detect_language(question)
    logger.info(f"Detected question language: {source_lang}")
    
    # Build query variants
    variants = [question]  # Always include original
    
    if source_lang == "de":
        # German question: add English variant
        en_variant = translate_query(question, "en")
        if en_variant:
            variants.append(en_variant)
    elif source_lang == "en":
        # English question: add German variant
        de_variant = translate_query(question, "de")
        if de_variant:
            variants.append(de_variant)
    elif source_lang == "fa":
        # Persian question: add German variant (primary doc language)
        de_variant = translate_query(question, "de")
        if de_variant:
            variants.append(de_variant)
    
    logger.info(f"Query variants: {len(variants)} ({', '.join([v[:30] + '...' for v in variants])})")
    
    # Retrieve with each variant (smaller k per variant)
    k_per_variant = 3
    all_results = []
    
    for i, variant in enumerate(variants, 1):
        try:
            logger.debug(
                "retrieval_variant_call: index=%s query_preview=%s",
                i,
                variant[:120],
            )
            results = similarity_search_with_score(
                variant,
                top_k=k_per_variant,
                where=where,
            )
            logger.debug(f"Variant {i} retrieved {len(results)} chunks")
            all_results.extend(results)
        except Exception as e:
            logger.warning(f"Retrieval failed for variant {i}: {e}")
            continue
    
    if not all_results:
        logger.warning("No results from any variant")
        return []
    
    # Deduplicate by chunk ID
    seen_ids = set()
    deduped = []
    
    for result in all_results:
        chunk_id = result.get('id', '')
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            deduped.append(result)
    
    logger.info(f"After deduplication: {len(deduped)} unique chunks")
    
    # Sort by score (lower is better for distance) and take top_k
    deduped.sort(key=lambda x: x.get('score', float('inf')))
    final_results = deduped[:top_k]
    
    logger.info(f"Returning top {len(final_results)} chunks")
    return final_results


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format Retrieved Chunks into Context String
    
    Builds a formatted context string from retrieved chunks,
    with source labels for reference.
    
    Args:
        retrieved_chunks: List of chunks with document text and metadata
    
    Returns:
        str: Formatted context string
    
    Example:
        [Source 1]
        Text content from first chunk...
        
        [Source 2]
        Text content from second chunk...
    """
    if not retrieved_chunks:
        return ""
    
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        # Extract text
        text = chunk.get("document", "").strip()
        
        # Add source label and text
        context_parts.append(f"[Source {i}]\n{text}")
    
    # Join with double newlines
    context = "\n\n".join(context_parts)
    
    logger.debug(f"Built context with {len(retrieved_chunks)} chunks, {len(context)} chars")
    
    return context


def build_citations(retrieved_chunks: List[Dict[str, Any]]) -> List[Citation]:
    """
    Build Citations from Retrieved Chunks
    
    Extracts citation information from chunk metadata.
    Creates Citation objects with source details and snippets.
    
    Args:
        retrieved_chunks: List of chunks with metadata
    
    Returns:
        List[Citation]: List of citation objects
    
    Citation includes:
        - source_id: Unique source identifier
        - source_type: Type (text/pdf/url)
        - source_uri: URI or filename
        - chunk_id: Chunk identifier
        - snippet: Text preview (200-400 chars)
        - score: Relevance score if available
    """
    citations = []
    
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        document = chunk.get("document", "")
        chunk_id = chunk.get("id", "")
        score = chunk.get("score")
        
        # Extract metadata fields with fallbacks
        source_id = metadata.get("source_id", "unknown")
        source_type = metadata.get("source_type", "unknown")
        source_name = metadata.get("source_name", "Unknown Source")
        
        # Create snippet (first 300 chars, max 400)
        snippet = document[:300].strip()
        if len(document) > 300:
            snippet += "..."
        
        # Create citation object
        citation = Citation(
            source_id=source_id,
            source_type=source_type,
            source_uri=source_name,
            chunk_id=chunk_id,
            snippet=snippet,
            score=score
        )
        
        citations.append(citation)
    
    logger.info(f"Built {len(citations)} citations")
    
    return citations


def _tokenize_text(value: str) -> set[str]:
    """Tokenize text into normalized alphanumeric word tokens."""

    if not value:
        return set()
    return set(re.findall(r"[a-zA-Z0-9äöüßÄÖÜ]+", value.lower()))


def _derive_page_key(metadata: Dict[str, Any], citation: Citation) -> str:
    """Build a stable dedupe key by source and page-like metadata if available."""

    for field_name in ("page", "page_number", "page_index", "chunk_index"):
        field_value = metadata.get(field_name)
        if field_value is not None:
            return str(field_value)
    return citation.chunk_id or "unknown"


def _is_single_fact_answer(answer: str) -> bool:
    """Heuristic for short single-fact answers."""

    answer_lower = answer.lower()
    numbers = set(re.findall(r"\b\d+\b", answer_lower))
    time_units = {
        "day", "days", "tag", "tage", "week", "weeks", "monat", "monate", "month", "months"
    }
    has_time_unit = any(unit in answer_lower for unit in time_units)
    has_single_numeric_time_statement = len(numbers) == 1 and has_time_unit
    return len(answer) < 240 or has_single_numeric_time_statement


def select_evidence_citations(
    user_question: str,
    generated_answer: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]],
    citations: List[Citation],
) -> List[Citation]:
    """Select minimal deterministic evidence citations from retrieved candidates."""

    if not citations:
        return []

    answer_tokens = _tokenize_text(generated_answer)
    question_tokens = _tokenize_text(user_question)
    numbers_in_answer = set(re.findall(r"\b\d+\b", generated_answer.lower()))
    priority_terms = {
        "day", "days", "tag", "tage", "register", "registration", "anmelden", "anmeldung"
    }
    key_terms = numbers_in_answer | (answer_tokens & priority_terms) | (question_tokens & priority_terms)

    candidate_rows: List[Tuple[float, int, float, Citation, str]] = []
    for index, citation in enumerate(citations):
        chunk_text = citation.snippet or ""
        retrieval_score = citation.score if citation.score is not None else float("inf")
        page_key = citation.chunk_id or "unknown"

        if retrieved_chunks and index < len(retrieved_chunks):
            chunk = retrieved_chunks[index]
            chunk_text = chunk.get("document", "") or chunk_text
            metadata = chunk.get("metadata", {}) or {}
            retrieval_score = chunk.get("score", retrieval_score)
            page_key = _derive_page_key(metadata, citation)

        if retrieval_score is None:
            retrieval_score = float("inf")

        chunk_tokens = _tokenize_text(chunk_text)
        key_term_hits = sum(1 for term in key_terms if term in chunk_tokens)
        overlap_count = len(answer_tokens & chunk_tokens)
        overlap_ratio = overlap_count / max(len(answer_tokens), 1)

        support_score = (key_term_hits * 5.0) + overlap_ratio
        candidate_rows.append((support_score, index, retrieval_score, citation, page_key))

    candidate_rows.sort(key=lambda row: (-row[0], row[1], row[2]))

    max_citations = 1 if _is_single_fact_answer(generated_answer) else 2
    selected: List[Citation] = []
    seen_source_page: set[Tuple[str, str]] = set()

    for _, _, _, citation, page_key in candidate_rows:
        dedupe_key = (citation.source_id, page_key)
        if dedupe_key in seen_source_page:
            continue
        seen_source_page.add(dedupe_key)
        selected.append(citation)
        if len(selected) >= max_citations:
            break

    if not selected:
        return citations[:1]

    return selected


def route_tools(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    use_policy: bool = True
) -> List[Dict[str, Any]]:
    """
    Route and Execute Tools Based on Question
    
    Analyzes the question and context to determine which tools to call.
    Now uses comprehensive routing policy (v2.0) with backward compatibility.
    
    Tool Routing Logic:
    - If use_policy=True (default): Uses comprehensive policy from tool_policy.py
    - If use_policy=False: Falls back to legacy heuristic routing
    
    Policy features (v2.0):
    - Intent detection with priority rules
    - Clarification handling (returns empty list if clarification needed)
    - Inline text vs RAG prioritization
    - Tool-first routing for deterministic tasks
    
    Args:
        question: User's question
        retrieved_chunks: Retrieved context chunks
        use_policy: Use comprehensive policy (default: True)
    
    Returns:
        List[ToolResult]: Results from tool calls
        Empty list if clarification needed (check logs for details)
    
    Example:
        >>> route_tools("Start date 01.03.2026 add 14 days", [])
        [ToolResult(tool_name="calculate_deadline", ...)]
    """
    logger.debug(
        "Tool routing is disabled: skipping tools for question '%s'", question[:80]
    )
    return []


def build_system_prompt(has_context: bool, response_language: str = "auto", detected_lang: str = "en") -> str:
    """
    Build System Prompt for LLM
    
    Creates appropriate system instruction based on whether
    context is available and desired response language.
    
    Args:
        has_context: Whether retrieved context is available
        response_language: Desired response language ("auto", "de", "en", "fa")
        detected_lang: Detected language of user's question (for auto mode)
    
    Returns:
        str: System prompt for LLM
    """
    
    # Determine response language
    if response_language == "auto":
        response_language = detected_lang
    
    # Language-specific instruction
    lang_instruction = ""
    if response_language == "de":
        lang_instruction = "\n\nIMPORTANT: Respond in German (Deutsch)."
    elif response_language == "en":
        lang_instruction = "\n\nIMPORTANT: Respond in English."
    elif response_language == "fa":
        lang_instruction = "\n\nIMPORTANT: Respond in Persian (فارسی). Use right-to-left script."
    
    if has_context:
        return f"""You are a helpful assistant specializing in German bureaucracy and immigration processes.

Answer the user's question using ONLY the information provided in the context below.{lang_instruction}

Guidelines:
- Be clear, precise, and helpful
- Reference specific information from the sources
- If the context doesn't contain enough information to fully answer the question, say so clearly
- Focus on German bureaucratic processes (Ausländerbehörde, visa, residence permits, etc.)
- Use a professional but friendly tone

If the question cannot be answered with the provided context, respond:
"I don't have enough information in the knowledge base to answer this question. Please ingest relevant documents first (e.g., official letters, forms, or immigration guidance)."
"""
    else:
        return f"""You are a helpful assistant specializing in German bureaucracy and immigration processes.

However, there are currently NO documents in the knowledge base.{lang_instruction}

Please inform the user that they need to ingest documents first before asking questions.
Suggest they upload:
- Official letters from Ausländerbehörde
- Visa or residence permit documents
- Immigration forms or guidance
- URLs to official German immigration websites
"""


def generate_answer(
    question: str,
    top_k: int = 5,
    chat_history: Optional[List[Dict[str, str]]] = None,
    response_language: str = "auto",
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate Answer Using RAG Pipeline
    
    Main RAG function that:
    1. Retrieves relevant chunks from vector store
    2. Builds context from chunks
    3. Generates answer using LLM
    4. Returns answer with citations
    
    Args:
        question: User's question
        top_k: Number of chunks to retrieve (default: 5)
        chat_history: Previous conversation messages (optional)
            Format: [{"role": "user"/"assistant", "content": "..."}]
        response_language: Target language for response ("auto", "de", "en", "fa")
            "auto" = same language as question
    
    Returns:
        Dict with keys:
            - answer: Generated answer text
            - citations: List of Citation objects
            - retrieved_count: Number of chunks retrieved
    
    Raises:
        ValueError: If question is empty or invalid
        Exception: If LLM call fails
    
    Example:
        result = generate_answer("What is the visa deadline?", top_k=5)
        print(result["answer"])
        for citation in result["citations"]:
            print(f"Source: {citation.source_uri}")
    """
    # Validate input
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    question = question.strip()
    logger.info(f"Generating answer for question: '{question[:100]}...'")
    logger.debug("generate_answer_start: top_k=%s response_language=%s", top_k, response_language)
    
    # Resolve model first so logs and pricing consistently reflect actual generation model.
    generation_model = _resolve_generation_model(model_override)

    # Detect question language for auto response mode
    detected_lang = detect_language(question)
    logger.info(f"Detected question language: {detected_lang}")
    
    # Step 1: Retrieve relevant chunks
    try:
        retrieved_chunks = multi_pass_retrieval(question, top_k=top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise ValueError(f"Failed to retrieve documents: {str(e)}")

    logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
    
    # Step 2: Build context and citations
    context = format_context(retrieved_chunks)
    citations = build_citations(retrieved_chunks)
    
    has_context = len(retrieved_chunks) > 0
    
    # Step 3: Build prompt with system instruction and context
    if has_context:
        system_prompt = build_system_prompt(has_context, response_language, detected_lang)
        user_prompt = f"""Context from knowledge base:

{context}

---

Question: {question}

Please provide a helpful answer based on the context above."""
    else:
        # No context
        system_prompt = build_system_prompt(False, response_language, detected_lang)
        user_prompt = question
    
    # Step 4: Include chat history (last 6 messages only)
    messages = [{"role": "system", "content": system_prompt}]
    
    if chat_history:
        # Include only last 6 messages to avoid token limits
        recent_history = chat_history[-6:]
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})
    
    # Add current question
    messages.append({"role": "user", "content": user_prompt})
    
    # Step 5: Call OpenAI LLM
    try:
        logger.debug(
            "llm_generate_call: model=%s messages=%s last_user_chars=%s",
            generation_model,
            len(messages),
            len(user_prompt),
        )
        llm = ChatOpenAI(
            model=generation_model,
            temperature=0.3,  # Low temperature for factual responses
            openai_api_key=settings.openai_api_key
        )
        
        response = llm.invoke(messages)
        answer = response.content.strip()
        token_usage = _extract_token_usage(response)
        estimated_cost_usd = _estimate_cost_usd(
            generation_model,
            token_usage["prompt_tokens"],
            token_usage["completion_tokens"],
        )

        evidence_citations = select_evidence_citations(
            user_question=question,
            generated_answer=answer,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
        )
        logger.info(
            "citation_evidence_selected: retrieved=%s evidence=%s",
            len(citations),
            len(evidence_citations),
        )
        
        logger.info(f"Generated answer: {len(answer)} chars")
        logger.debug("llm_generate_result_preview: %s", answer[:300])
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        raise Exception(f"Failed to generate answer: {str(e)}")
    
    # Step 6: Return result
    return {
        "answer": answer,
        "citations": evidence_citations,
        "retrieved_count": len(retrieved_chunks),
        "tool_results": None,
        "model_used": generation_model,
        "token_usage": token_usage,
        "estimated_cost_usd": estimated_cost_usd,
    }
