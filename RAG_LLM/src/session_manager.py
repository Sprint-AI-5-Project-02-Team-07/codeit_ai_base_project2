import os
import json
import config

SESSION_FILE = os.path.join(config.DATA_DIR, ".session_context.json")

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_session(context):
    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save session context: {e}")

def update_context(new_filters, last_query):
    context = load_session()
    
    # Update agency if present, otherwise keep previous (sticky context)
    # Strategy: If new filters have agency, overwrite. If not, keep old.
    if new_filters.get('agency'):
        context['agency'] = new_filters['agency']
    
    if new_filters.get('min_amount'):
        context['min_amount'] = new_filters['min_amount']
        
    context['last_query'] = last_query
    save_session(context)
    return context

def get_merged_filters(current_filters):
    context = load_session()
    merged = current_filters.copy()
    
    # Check for Explicit Reset
    if current_filters.get('reset_context'):
        print("Tip: Context Reset triggered by user query.")
        # Do not inherit agency
        # We should also clear it from the session file for future turns?
        # Yes, update the session file to remove agency
        if context.get('agency'):
            del context['agency']
            save_session(context)
        return merged
    
    # If current has no agency, use context's agency
    if not merged.get('agency') and context.get('agency'):
        print(f"Tip: Applying context from previous turn (Agency: {context['agency']})")
        merged['agency'] = context['agency']
        
    return merged
