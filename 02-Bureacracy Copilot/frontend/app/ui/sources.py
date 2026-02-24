"""Sources panel UI for ingestion queue and knowledge base management."""

import streamlit as st
import time
from datetime import datetime
from math import ceil


def _build_source_name(raw_name: str, fallback_prefix: str = "Text Note") -> str:
    """Return a clean source name, generating a fallback when input is blank.

    Args:
        raw_name (str): User-provided source name from UI input.
        fallback_prefix (str): Prefix used when generating fallback names.

    Returns:
        str: Non-empty source name safe for ingestion requests.
    """
    if raw_name and raw_name.strip():
        return raw_name.strip()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"{fallback_prefix} {timestamp}"


def _extract_ingest_summary(response) -> dict:
    """Extract normalized ingestion summary fields from API response payload.

    Args:
        response: API response object returned by ingestion methods.

    Returns:
        dict: Summary fields (`source_id`, `source_name`, `chunks`, `characters`, `added_at`).
    """
    data = getattr(response, "data", {}) or {}
    source = data.get("source", {}) if isinstance(data, dict) else {}

    return {
        "source_id": source.get("id", "-"),
        "source_name": source.get("uri", "-"),
        "chunks": source.get("chunks", 0),
        "characters": source.get("characters", 0),
        "added_at": source.get("added_at", "-"),
    }


def _render_ingest_success(summary: dict) -> None:
    """Render compact and detailed success feedback for one ingestion action.

    Args:
        summary (dict): Normalized summary dictionary from `_extract_ingest_summary`.

    Returns:
        None: Writes success message and optional detail rows to UI.
    """
    st.success("Ingestion completed")
    st.caption(
        f"Chunks: {summary['chunks']} · Characters: {summary['characters']}"
    )

    with st.expander("Ingestion details", expanded=False):
        st.markdown(f"**Source:** {summary['source_name']}")
        st.markdown(f"**Source ID:** {summary['source_id']}")
        st.markdown(f"**Chunks:** {summary['chunks']}")
        st.markdown(f"**Characters:** {summary['characters']}")
        st.markdown(f"**Added at:** {summary['added_at']}")

def _queue_item(item_type, data, name):
    """Append a source item to the pending ingestion queue.

    Args:
        item_type (str): Queue item type such as `text`, `pdf`, or `url`.
        data: Raw payload associated with the queued item.
        name (str): Human-readable name shown in the UI queue.

    Returns:
        None: Updates `st.session_state.queue` in place.
    """
    if "queue" not in st.session_state:
        st.session_state.queue = []
    
    st.session_state.queue.append(
        {
            "type": item_type,
            "data": data,
            "name": name,
            "added_at": datetime.now().isoformat(),
        }
    )

def _ingest_item(client, item):
    """Dispatch one queued item to the correct backend ingestion method.

    Args:
        client: Frontend API client instance.
        item (dict): Queue record containing `type`, `data`, and `name` fields.

    Returns:
        object: API response object with `success`/`error` semantics.
    """
    if item["type"] == "text":
        return client.ingest_text(item["data"], source_name=item["name"])
    elif item["type"] == "pdf":
        return client.ingest_pdf(item["data"], filename=item["name"])
    elif item["type"] == "url":
        return client.ingest_url(item["data"])
    else:
        # Mock failure response if type unknown
        class MockResponse:
            success = False
            error = "Unsupported type"
        return MockResponse()

def render(client):
    """Render ingestion controls, queue processing, and KB source list.

    Args:
        client: Frontend API client used for ingestion and source queries.

    Returns:
        None: Writes left-panel Streamlit controls and handles user actions.
    """
    # Step 1: render source creation controls (text/file/url) and queue actions.
    # Render panel title so users can identify ingestion workspace quickly.
    st.subheader("Sources")

    # Keep a compact "last ingest" summary visible for transparency after each action.
    if "last_ingest_summary" not in st.session_state:
        st.session_state.last_ingest_summary = None

    if st.session_state.last_ingest_summary:
        last = st.session_state.last_ingest_summary
        st.info(
            f"Last ingest · {last['source_name']} · {last['chunks']} chunks · {last['characters']} chars"
        )

    # --- Add Source UI ---
    if "show_add_source" not in st.session_state:
        st.session_state.show_add_source = False

    # Render toggle button that expands/collapses the source input section.
    if st.button("+ Add Source", key="btn_toggle_add_source"):
        st.session_state.show_add_source = not st.session_state.show_add_source

    if st.session_state.show_add_source:
        # Group all add-source controls in one container for consistent spacing.
        with st.container():
            st.markdown("#### New Source")
            # Render tabs so each ingestion mode has an isolated input flow.
            tab_text, tab_file, tab_url = st.tabs(["Text", "File", "URL"])

            # A) Paste Text
            with tab_text:
                # Collect free-form text content to send through ingest_text endpoint.
                txt_input = st.text_area("Content", height=100, key="ingest_text_area")
                # Collect explicit source name so retrieval/source list remains human-readable.
                txt_name = st.text_input("Source Name", key="ingest_text_name", placeholder="e.g. Meeting Notes")
                st.caption("Source name is optional. A default name is generated if left empty.")

                # Show clean pre-ingestion metrics to help users estimate processing scope.
                input_chars = len(txt_input.strip()) if txt_input else 0
                estimated_chunks = ceil(input_chars / 1000) if input_chars > 0 else 0
                st.caption(f"Preview · Characters: {input_chars} · Estimated chunks: {estimated_chunks}")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Add to Queue", key="btn_text_queue", use_container_width=True):
                        if txt_input and txt_input.strip():
                            queue_name = _build_source_name(txt_name, fallback_prefix="Text Note")
                            # Queue item instead of immediate ingestion to support batch processing.
                            _queue_item("text", txt_input, queue_name)
                            st.success("Added to queue")
                        else:
                            st.error("Missing content")
                with c2:
                    if st.button("Ingest Now", key="btn_text_ingest", use_container_width=True):
                        if txt_input and txt_input.strip():
                            source_name = _build_source_name(txt_name, fallback_prefix="Text Note")
                            with st.spinner("Ingesting..."):
                                try:
                                    # Call backend text-ingestion endpoint through API client abstraction.
                                    response = client.ingest_text(txt_input, source_name=source_name)
                                    if response.success:
                                        summary = _extract_ingest_summary(response)
                                        st.session_state.last_ingest_summary = summary
                                        _render_ingest_success(summary)
                                        time.sleep(1)
                                        st.session_state.show_add_source = False
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {response.error}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("Missing content")

            # B) Upload File
            with tab_file:
                # Accept supported file types that map directly to backend ingestion methods.
                uploaded_file = st.file_uploader("PDF or TXT", type=["pdf", "txt"], key="ingest_file_uploader")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Add to Queue", key="btn_file_queue", use_container_width=True):
                        if uploaded_file:
                            if uploaded_file.type == "application/pdf":
                                data = uploaded_file.getvalue()
                                # Keep original file name so users can map queue entry to source list.
                                _queue_item("pdf", data, uploaded_file.name)
                            else:
                                data = uploaded_file.getvalue().decode("utf-8")
                                _queue_item("text", data, uploaded_file.name)
                            st.success("Added to queue")
                with c2:
                    if st.button("Ingest Now", key="btn_file_ingest", use_container_width=True):
                        if uploaded_file:
                            with st.spinner("Uploading & Ingesting..."):
                                try:
                                    if uploaded_file.type == "application/pdf":
                                        pdf_bytes = uploaded_file.getvalue()
                                        # Route PDF bytes to dedicated backend PDF ingestion endpoint.
                                        response = client.ingest_pdf(pdf_bytes, filename=uploaded_file.name)
                                    else:
                                        # Text file
                                        text = uploaded_file.getvalue().decode("utf-8")
                                        # Route text files through text ingestion endpoint for consistency.
                                        response = client.ingest_text(text, source_name=uploaded_file.name)
                                    
                                    if response.success:
                                        summary = _extract_ingest_summary(response)
                                        st.session_state.last_ingest_summary = summary
                                        _render_ingest_success(summary)
                                        time.sleep(1)
                                        st.session_state.show_add_source = False
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {response.error}")
                                except Exception as e:
                                    st.error(f"Error: {e}")

            # C) URL
            with tab_url:
                # Capture URL to be fetched and ingested by backend/web extraction pipeline.
                url_input = st.text_input("URL", key="ingest_url_input")
                # URL ingestion endpoint doesn't support custom source name currently
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Add to Queue", key="btn_url_queue", use_container_width=True):
                        if url_input:
                            # Queue URL so users can process multiple items in one batch.
                            _queue_item("url", url_input, url_input)
                            st.success("Added to queue")
                with c2:
                    if st.button("Ingest Now", key="btn_url_ingest", use_container_width=True):
                        if url_input:
                            with st.spinner("Scraping..."):
                                try:
                                    # Call backend URL ingestion endpoint to fetch and chunk remote content.
                                    response = client.ingest_url(url_input)
                                    if response.success:
                                        summary = _extract_ingest_summary(response)
                                        st.session_state.last_ingest_summary = summary
                                        _render_ingest_success(summary)
                                        time.sleep(1)
                                        st.session_state.show_add_source = False
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {response.error}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
        st.markdown("---")

    # Step 2: render queue state and batch-processing controls.
    # --- Queue ---
    if "queue" not in st.session_state:
        st.session_state.queue = []
    
    st.markdown("**Queue**")
    
    if st.session_state.queue:
        # Render queue items
        # Iterate queue items to provide visibility and item-level removal controls.
        for idx, item in enumerate(st.session_state.queue):
            item_type = item.get("type", "?")
            icon = {"pdf": "📄", "text": "📝", "url": "🌐"}.get(item_type, "📦")
            name = item.get("name", "Unknown")
            
            c_name, c_del = st.columns([0.8, 0.2])
            with c_name:
                st.markdown(f"{icon} {name[:30]}")
            with c_del:
                if st.button("✕", key=f"del_queue_{idx}", help="Remove"):
                    # Remove selected queue entry immediately and rerun to refresh widget state.
                    st.session_state.queue.pop(idx)
                    st.rerun()
        
        # Action Buttons
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Process Queue", key="btn_process_queue", use_container_width=True, type="primary"):
                total = len(st.session_state.queue)
                progress_bar = st.progress(0)
                errors = []
                success_count = 0
                
                # Iterate over a copy so button-driven state updates do not mutate
                # the list we are currently traversing.
                queue_copy = list(st.session_state.queue)
                
                for idx, item in enumerate(queue_copy):
                    try:
                        # Delegate type-specific ingestion routing to helper for centralized behavior.
                        resp = _ingest_item(client, item)
                        if resp.success:
                            success_count += 1
                        else:
                            errors.append(f"{item['name']}: {resp.error}")
                    except Exception as e:
                        errors.append(f"{item['name']}: {str(e)}")
                    
                    # Advance visual progress so users can track batch completion in real time.
                    progress_bar.progress((idx + 1) / total)
                
                if success_count == total:
                    st.session_state.queue = []
                    st.success("All items processed!")
                    time.sleep(1)
                    st.rerun()
                else:
                    # Update queue to only keep failed items ?? Or just show errors?
                    # For simplicity, clear queue of successful ones if we tracked indices, 
                    # but here we just show report.
                    st.warning(f"Processed {success_count}/{total} items.")
                    st.error("Some queue items failed. See details below:")
                    for e in errors:
                        st.error(e)
                    # Clear queue if user wants
        
        with c2:
            if st.button("Clear Queue", key="btn_clear_queue", use_container_width=True):
                # Clear queue explicitly on user request.
                st.session_state.queue = []
                st.rerun()
                
    else:
        st.caption("No items in queue")
    
    st.markdown("---")

    # Step 3: fetch and render persisted knowledge-base sources plus UI-only selection toggles.
    # --- Knowledge Base List ---
    # Fetch sources
    try:
        # Pull latest source inventory from backend so count/cards reflect current persisted state.
        response = client.get_sources()
        if response.success:
            sources_list = response.data.get('sources', [])
        else:
            sources_list = []
    except:
        sources_list = []
    
    col_kb_title, col_kb_clear = st.columns([0.7, 0.3])
    with col_kb_title:
        st.markdown(f"**Knowledge Base ({len(sources_list)})**")
    with col_kb_clear:
        if sources_list:
            if st.button("Clear", key="btn_clear_kb", type="primary", help="Delete all documents"):
                try:
                    # Trigger backend clear endpoint to remove all persisted vector-store documents.
                    res = client.clear_knowledge_base()
                    if res.success:
                        st.success("Cleared!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to clear")
                except Exception as e:
                    st.error(f"Error: {e}")

    for idx, src in enumerate(sources_list):
        # source object shape based on CLI output: {'source_type': '...', 'uri': '...', 'chunk_count': ...}
        # backend might return 'id' or 'source_id'
        s_id = src.get("id") or src.get("source_id") or str(idx)
        
        # Name resolution: try source_name -> name -> uri -> source -> Untitled
        s_name = src.get("source_name") or src.get("name") or src.get("uri") or src.get("source") or "Untitled"
        
        # UI Card
        with st.container():
            with st.container():
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-card-title">{s_name}</div>
                    <div class="source-card-meta">ID: {str(s_id)[:8]}...</div>
                </div>
                """, unsafe_allow_html=True)