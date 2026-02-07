import streamlit as st
import psycopg2
import pandas as pd
from datetime import date
from typing import Any, Dict, Optional, Tuple
import os
import json
import requests

import streamlit as st

st.set_page_config(
    page_title="Rule Conflict Database",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    /* Kill Streamlit's mobile sidebar toggle text/icons */
    button[kind="header"] {
        display: none !important;
    }

    /* Extra safety for newer Streamlit builds */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Hide stray material icon text */
    span[class*="material-icons"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432"),
        sslmode="require"
    )


def execute_non_query(query: str, params: Tuple[Any, ...]) -> None:
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
    except Exception as exc:
        if conn is not None:
            conn.rollback()
        raise exc
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def fetch_dataframe(query: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:

    conn = None
    try:
        conn = get_connection()
        if params:
            return pd.read_sql(query, conn, params=params)
        return pd.read_sql(query, conn)
    finally:
        if conn is not None:
            conn.close()


# Hugging Face: router-based free inference; key from env only (no hardcoding)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACE_INFERENCE_URL = f"https://router.huggingface.co/hf-inference/models/{HUGGINGFACE_MODEL}"
HUGGINGFACE_TIMEOUT = 45

def _get_conflict_field(row: Dict[str, Any], *keys: str, default: str = "") -> str:
    """Get first present key from conflict row; normalize to str and strip. Ignores NaN."""
    for k in keys:
        v = row.get(k)
        if v is None or (hasattr(pd, "isna") and pd.isna(v)):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return default


def explain_conflict_ai(conflict: dict) -> str:
    """
    Call Hugging Face router inference API for a conflict-specific, analytical explanation.
    Does not modify DB, resolve conflicts, run SQL, or change priorities.
    Returns error-style string (e.g. leading '*') on failure so caller can use fallback.
    """
    rule_1_name = _get_conflict_field(conflict, "rule_1_name", "rule_name_1", default="Rule 1")
    rule_2_name = _get_conflict_field(conflict, "rule_2_name", "rule_name_2", default="Rule 2")
    rule_category_1 = _get_conflict_field(conflict, "rule_category_1", "category_1", default="(not specified)")
    rule_category_2 = _get_conflict_field(conflict, "rule_category_2", "category_2", default="(not specified)")
    action_1 = _get_conflict_field(conflict, "action_1", "action_type_1", default="action 1")
    action_2 = _get_conflict_field(conflict, "action_2", "action_type_2", default="action 2")
    target_entity = _get_conflict_field(conflict, "target_entity", "target", default="target")
    conflict_reason = _get_conflict_field(conflict, "conflict_reason", "reason", default="Conflict detected.")
    active_from_1 = _get_conflict_field(conflict, "active_from_1", "active_from_rule_1", default="")
    active_to_1 = _get_conflict_field(conflict, "active_to_1", "active_to_rule_1", default="")
    active_from_2 = _get_conflict_field(conflict, "active_from_2", "active_from_rule_2", default="")
    active_to_2 = _get_conflict_field(conflict, "active_to_2", "active_to_rule_2", default="")
    range_1 = f"{active_from_1} to {active_to_1}" if active_from_1 or active_to_1 else "(not specified)"
    range_2 = f"{active_from_2} to {active_to_2}" if active_from_2 or active_to_2 else "(not specified)"

    prompt = (
        "You are analyzing a specific rule conflict. Do NOT give generic explanations. "
        "Tailor every sentence to the rule names, categories, and actions below. Do not repeat boilerplate.\n\n"
        "--- Section 1: Conflict Context ---\n"
        f"Rule 1: {rule_1_name} (category: {rule_category_1}). Active: {range_1}.\n"
        f"Rule 2: {rule_2_name} (category: {rule_category_2}). Active: {range_2}.\n"
        f"Target entity: {target_entity}.\n"
        f"Database conflict reason: {conflict_reason}\n\n"
        "--- Section 2: Condition Analysis ---\n"
        "Based on the context above: can the conditions of BOTH rules be true for the same entity at the same time? "
        "Analyze why or how they overlap (refer to the specific rules by name).\n\n"
        "--- Section 3: Action Contradiction ---\n"
        f"Rule 1 action: {action_1}. Rule 2 action: {action_2}. "
        "Explain how these two actions oppose each other and what real-world effect this contradiction causes for the target entity.\n\n"
        "--- Section 4: Why This Conflict Happens ---\n"
        "Answer in one clear sentence: Why can a single real-world entity satisfy both rules at once? "
        "Be specific to these rules, not generic.\n\n"
        "--- Section 5: Resolution Ideas ---\n"
        "Give exactly 2–3 resolution strategies that are specific to THIS conflict (refer to the rule names and the overlap you identified). "
        "No code or SQL. No generic advice."
    )

    if not HUGGINGFACE_API_KEY:
        return "*AI explanation unavailable: `HUGGINGFACE_API_KEY` is not set.*"

    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 400, "temperature": 0.4},
        }
        resp = requests.post(
            HUGGINGFACE_INFERENCE_URL,
            headers=headers,
            json=payload,
            timeout=HUGGINGFACE_TIMEOUT,
        )
        if resp.status_code != 200:
            try:
                err = resp.json().get("error", resp.text[:200])
            except Exception:
                err = resp.text[:200] if resp.text else str(resp.status_code)
            return f"*API returned {resp.status_code}: {err}*"
        data = resp.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict):
            if "error" in data:
                return f"*API error: {data['error']}*"
            if "generated_text" in data:
                return str(data["generated_text"]).strip()
        return "*Could not parse AI response.*"
    except requests.exceptions.Timeout:
        return "*Request timed out. Try again or use the fallback explanation.*"
    except requests.exceptions.RequestException as e:
        return f"*Request failed: {str(e)[:200]}*"
    except Exception as e:
        return f"*Unexpected error: {str(e)[:200]}*"


def explain_conflict_fallback(conflict: dict) -> str:
    """
    Deterministic, human-readable explanation and resolution ideas when AI is
    unavailable or rate-limited. Does not modify data or resolve conflicts.
    """
    rule_1_name = _get_conflict_field(conflict, "rule_1_name", "rule_name_1", default="Rule 1")
    rule_2_name = _get_conflict_field(conflict, "rule_2_name", "rule_name_2", default="Rule 2")
    action_1 = _get_conflict_field(conflict, "action_1", "action_type_1", default="action 1")
    action_2 = _get_conflict_field(conflict, "action_2", "action_type_2", default="action 2")
    target_entity = _get_conflict_field(conflict, "target_entity", "target", default="target")
    conflict_reason = _get_conflict_field(conflict, "conflict_reason", "reason", default="Conflict detected.")

    return (
        f"**Explanation**\n\n"
        f"Rules **{rule_1_name}** and **{rule_2_name}** conflict on the same target (**{target_entity}**). "
        f"One specifies **{action_1}** and the other **{action_2}**, so the system cannot apply both. "
        f"The database reports: {conflict_reason}\n\n"
        "**Resolution ideas**\n\n"
        "1. **Adjust priorities** — Give one rule higher priority so it is applied first.\n"
        "2. **Narrow conditions** — Restrict one or both rules (e.g. by attribute or value) so they do not apply to the same cases.\n"
        "3. **Merge or remove** — If both intend the same outcome, consider merging; otherwise remove or disable one rule."
    )


def validate_user(username: str, password: str) -> Optional[Tuple[int, str]]:
    """
    Validate credentials against the users table.
    Returns (user_id, username) on success, None on failure.
    """
    query = "SELECT user_id, username FROM users WHERE username = %s AND password = %s"
    try:
        df = fetch_dataframe(query, (username.strip(), password))
        if df.empty or len(df) == 0:
            return None
        row = df.iloc[0]
        return (int(row["user_id"]), str(row["username"]))
    except Exception:
        return None


def login_page() -> None:
    st.subheader("Login")
    st.caption("Enter your credentials to access the Rule Conflict Database.")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if not username or not username.strip():
                st.error("Please enter a username.")
            elif not password:
                st.error("Please enter a password.")
            else:
                result = validate_user(username, password)
                if result is not None:
                    user_id, name = result
                    st.session_state.user_id = user_id
                    st.session_state.username = name
                    st.success("Login successful.")
                    st.toast(f"Welcome, {name}!", icon="✅")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")


def add_rule(rule_name: str, category: str, priority: int, active_from: date, active_to: date, user_id: int) -> None:
    query = """
        INSERT INTO rules (rule_name, rule_category, priority, active_from, active_to, user_id)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    params = (rule_name, category, priority, active_from, active_to, user_id)
    execute_non_query(query, params)


def add_condition(rule_id: int, attribute_name: str, operator: str, value: str) -> None:
    query = """
        INSERT INTO rule_conditions (rule_id, attribute_name, operator, value)
        VALUES (%s, %s, %s, %s)
    """
    params = (rule_id, attribute_name, operator, value)
    execute_non_query(query, params)


def add_action(rule_id: int, action_type: str, target_entity: str) -> None:
    query = """
        INSERT INTO rule_actions (rule_id, action_type, target_entity)
        VALUES (%s, %s, %s)
    """
    params = (rule_id, action_type, target_entity)
    execute_non_query(query, params)


def rule_status_exists(status_name: str) -> bool:
    """Return True if status_name already exists in rule_status."""
    df = fetch_dataframe("SELECT 1 AS ok FROM rule_status WHERE status_name = %s", (status_name.strip(),))
    return not df.empty


def add_rule_status(status_name: str, description: str) -> None:
    query = """
        INSERT INTO rule_status (status_name, description)
        VALUES (%s, %s)
    """
    params = (status_name.strip(), description.strip() if description else "")
    execute_non_query(query, params)


def add_rule_version(rule_id: int, version_number: str, rule_snapshot: str, is_active: bool) -> None:
    query = """
        INSERT INTO rule_versions (rule_id, version_number, rule_snapshot, is_active)
        VALUES (%s, %s, %s, %s)
    """
    params = (rule_id, version_number.strip(), rule_snapshot.strip(), is_active)
    execute_non_query(query, params)


def rule_priority_exists(rule_id: int) -> bool:
    """Return True if this rule already has a row in rule_priority (one priority per rule)."""
    df = fetch_dataframe("SELECT 1 AS ok FROM rule_priority WHERE rule_id = %s", (rule_id,))
    return not df.empty


def add_rule_priority(rule_id: int, priority_level: int) -> None:
    query = """
        INSERT INTO rule_priority (rule_id, priority_level)
        VALUES (%s, %s)
    """
    params = (rule_id, priority_level)
    execute_non_query(query, params)


def delete_rule(rule_id: int, user_id: int) -> None:
    """Delete a rule owned by the user. Related conditions and actions are deleted via CASCADE."""
    query = "DELETE FROM rules WHERE rule_id = %s AND user_id = %s"
    params = (rule_id, user_id)
    execute_non_query(query, params)


def delete_condition(condition_id: int) -> None:
    """Delete a specific condition."""
    query = "DELETE FROM rule_conditions WHERE condition_id = %s"
    params = (condition_id,)
    execute_non_query(query, params)


def delete_action(action_id: int) -> None:
    """Delete a specific action."""
    query = "DELETE FROM rule_actions WHERE action_id = %s"
    params = (action_id,)
    execute_non_query(query, params)


@st.cache_data(ttl=60)
def fetch_rules(user_id: int) -> pd.DataFrame:
    """Return available rules for selection (cached briefly for smoother UX)."""
    query = "SELECT rule_id, rule_name, rule_category, priority FROM rules WHERE user_id = %s ORDER BY rule_id"
    return fetch_dataframe(query, (user_id,))


@st.cache_data(ttl=60)
def fetch_conflicts(user_id: int) -> pd.DataFrame:
    """Return conflicts scoped to the user's rules."""
    query = "SELECT * FROM rule_conflicts_view"
    return fetch_dataframe(query)


@st.cache_data(ttl=60)
def fetch_all_rules(user_id: int) -> pd.DataFrame:
    """Return all rules for the overview tab."""
    query = "SELECT * FROM rules WHERE user_id = %s ORDER BY rule_id"
    return fetch_dataframe(query, (user_id,))


@st.cache_data(ttl=60)
def fetch_all_conditions(user_id: int) -> pd.DataFrame:
    """Return all rule conditions joined with rule names (user-scoped)."""
    query = """
        SELECT 
            rc.condition_id,
            rc.rule_id,
            r.rule_name,
            rc.attribute_name,
            rc.operator,
            rc.value
        FROM rule_conditions rc
        JOIN rules r ON rc.rule_id = r.rule_id AND r.user_id = %s
        ORDER BY rc.rule_id, rc.condition_id
    """
    return fetch_dataframe(query, (user_id,))


@st.cache_data(ttl=60)
def fetch_all_actions(user_id: int) -> pd.DataFrame:
    """Return all rule actions joined with rule names (user-scoped)."""
    query = """
        SELECT 
            ra.action_id,
            ra.rule_id,
            r.rule_name,
            ra.action_type,
            ra.target_entity
        FROM rule_actions ra
        JOIN rules r ON ra.rule_id = r.rule_id AND r.user_id = %s
        ORDER BY ra.rule_id, ra.action_id
    """
    return fetch_dataframe(query, (user_id,))


@st.cache_data(ttl=60)
def fetch_rules_for_delete(user_id: int) -> pd.DataFrame:
    """Return rules with rule_id and rule_name for delete dropdown."""
    query = "SELECT rule_id, rule_name FROM rules WHERE user_id = %s ORDER BY rule_id"
    return fetch_dataframe(query, (user_id,))


@st.cache_data(ttl=60)
def fetch_conditions_for_delete(user_id: int) -> pd.DataFrame:
    """Return all conditions with condition_id for deletion (user-scoped)."""
    query = """
        SELECT 
            rc.condition_id,
            rc.rule_id,
            r.rule_name,
            rc.attribute_name,
            rc.operator,
            rc.value
        FROM rule_conditions rc
        JOIN rules r ON rc.rule_id = r.rule_id AND r.user_id = %s
        ORDER BY rc.condition_id
    """
    return fetch_dataframe(query, (user_id,))


@st.cache_data(ttl=60)
def fetch_actions_for_delete(user_id: int) -> pd.DataFrame:
    """Return all actions with action_id for deletion (user-scoped)."""
    query = """
        SELECT 
            ra.action_id,
            ra.rule_id,
            r.rule_name,
            ra.action_type,
            ra.target_entity
        FROM rule_actions ra
        JOIN rules r ON ra.rule_id = r.rule_id AND r.user_id = %s
        ORDER BY ra.action_id
    """
    return fetch_dataframe(query, (user_id,))


def check_db_health() -> Tuple[bool, Optional[str]]:

    try:

        _ = fetch_dataframe("SELECT 1 AS ok")
        return True, None
    except Exception as exc:
        return False, str(exc)



def show_sidebar(user_id: int, username: str) -> None:

    st.sidebar.title("Rule Conflict DB")
    st.sidebar.caption("Manage rules, conditions, actions, and view conflicts.")
    st.sidebar.markdown(f"**Logged in as:** {username}")

    # Database status
    healthy, error_msg = check_db_health()
    if healthy:
        st.sidebar.success("Database connected", icon="✅")
    else:
        st.sidebar.error("Database error", icon="⚠️")
        with st.sidebar.expander("Details", expanded=False):
            st.code(error_msg or "Unknown error")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", type="primary"):
        for key in ("user_id", "username"):
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("All rule evaluation and conflict detection is implemented in SQL.")


def show_add_rule_page(user_id: int) -> None:
    st.subheader("Add Rule")
    st.caption("Create a new rule. Business logic and conflict detection remain in the database.")

    with st.form("add_rule_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            rule_name = st.text_input("Rule Name", placeholder="e.g., Scholarship Rule A")
            rule_category = st.text_input("Rule Category", placeholder="e.g., Scholarship, Admission")
            priority = st.number_input("Priority", min_value=1, step=1, value=1)

        with col2:
            active_from = st.date_input("Active From", value=date.today())
            active_to = st.date_input("Active To", value=date.today())

        submitted = st.form_submit_button("Add Rule")

        if submitted:
            if not rule_name.strip():
                st.warning("Please provide a rule name.")
            elif not rule_category.strip():
                st.warning("Please provide a rule category.")
            else:
                try:
                    add_rule(rule_name.strip(), rule_category.strip(), int(priority), active_from, active_to, user_id)
                    st.success("Rule added successfully.")
                    st.toast("Rule added.", icon="✅")
                    # Clear rule cache so new rule appears in dropdowns and overview
                    fetch_rules.clear()
                    fetch_all_rules.clear()
                    fetch_rules_for_delete.clear()
                except Exception as exc:
                    st.error("Unable to add rule. Please check the database connection and constraints.")
                    with st.expander("Error details"):
                        st.code(str(exc))


def show_add_condition_page(user_id: int) -> None:
    st.subheader("Add Rule Condition")
    st.caption("Attach a condition to an existing rule (owned by you).")

    try:
        rules_df = fetch_rules(user_id)
    except Exception as exc:
        st.error("Unable to load rules from the database.")
        with st.expander("Error details"):
            st.code(str(exc))
        return

    if rules_df.empty:
        st.info("No rules found. Please add a rule first.")
        return

    # Build a friendly label -> id mapping
    options: Dict[str, int] = {}
    for _, row in rules_df.iterrows():
        label = f"[{row['rule_id']}] {row['rule_name']} (Priority {row['priority']})"
        options[label] = int(row["rule_id"])

    with st.form("add_condition_form", clear_on_submit=False):
        selected_label = st.selectbox("Select Rule", options=list(options.keys()))
        attribute_name = st.text_input("Attribute Name", placeholder="e.g., gpa, year, department")
        operator = st.selectbox("Operator", [">", ">=", "<", "<=", "=", "!=", "IN", "NOT IN"])
        value = st.text_input("Value", placeholder="e.g., 8.0, 'CSE', (1, 2, 3)")

        submitted = st.form_submit_button("Add Condition")

        if submitted:
            if not attribute_name.strip():
                st.warning("Please provide an attribute name.")
            elif not value.strip():
                st.warning("Please provide a value.")
            else:
                rule_id = options[selected_label]
                try:
                    # All interpretation of attribute/operator/value is done in SQL
                    add_condition(rule_id, attribute_name.strip(), operator, value.strip())
                    st.success("Condition added successfully.")
                    st.toast("Condition added.", icon="✅")
                    # Clear cache so new condition appears in overview
                    fetch_all_conditions.clear()
                    fetch_conditions_for_delete.clear()
                except Exception as exc:
                    st.error("Unable to add condition. Please check the database and constraints.")
                    with st.expander("Error details"):
                        st.code(str(exc))


def show_add_action_page(user_id: int) -> None:
    st.subheader("Add Rule Action")
    st.caption("Attach an action to an existing rule (owned by you).")

    try:
        rules_df = fetch_rules(user_id)
    except Exception as exc:
        st.error("Unable to load rules from the database.")
        with st.expander("Error details"):
            st.code(str(exc))
        return

    if rules_df.empty:
        st.info("No rules found. Please add a rule first.")
        return

    options: Dict[str, int] = {}
    for _, row in rules_df.iterrows():
        label = f"[{row['rule_id']}] {row['rule_name']} (Priority {row['priority']})"
        options[label] = int(row["rule_id"])

    with st.form("add_action_form", clear_on_submit=False):
        selected_label = st.selectbox("Select Rule", options=list(options.keys()))
        action_type = st.selectbox("Action Type", ["allow", "deny"])
        target_entity = st.text_input("Target Entity", placeholder="e.g., student, application, transaction")

        submitted = st.form_submit_button("Add Action")

        if submitted:
            if not target_entity.strip():
                st.warning("Please provide a target entity.")
            else:
                rule_id = options[selected_label]
                try:
                    add_action(rule_id, action_type, target_entity.strip())
                    st.success("Action added successfully.")
                    st.toast("Action added.", icon="✅")
                    # Clear cache so new action appears in overview
                    fetch_all_actions.clear()
                    fetch_actions_for_delete.clear()
                except Exception as exc:
                    st.error("Unable to add action. Please check the database and constraints.")
                    with st.expander("Error details"):
                        st.code(str(exc))


def show_add_rule_status_page() -> None:
    st.subheader("Add Rule Status")
    st.caption("Insert a new status into rule_status. Duplicate status names are not allowed.")

    with st.form("add_rule_status_form", clear_on_submit=False):
        status_name = st.text_input("Status Name", placeholder="e.g., active, draft, retired")
        description = st.text_area("Description", placeholder="Optional description")
        submitted = st.form_submit_button("Add Rule Status")

        if submitted:
            if not status_name or not status_name.strip():
                st.warning("Please provide a status name.")
            else:
                if rule_status_exists(status_name):
                    st.error("A status with this name already exists. Choose a different status name.")
                else:
                    try:
                        add_rule_status(status_name.strip(), description.strip() if description else "")
                        st.success("Rule status added successfully.")
                        st.toast("Rule status added.", icon="✅")
                    except Exception as exc:
                        st.error("Unable to add rule status. Please check the database and constraints.")
                        with st.expander("Error details"):
                            st.code(str(exc))


def show_add_rule_version_page(user_id: int) -> None:
    st.subheader("Add Rule Version")
    st.caption("Insert a version snapshot for an existing rule.")

    try:
        rules_df = fetch_rules(user_id)
    except Exception as exc:
        st.error("Unable to load rules from the database.")
        with st.expander("Error details"):
            st.code(str(exc))
        return

    if rules_df.empty:
        st.info("No rules found. Please add a rule first.")
        return

    options: Dict[str, int] = {}
    for _, row in rules_df.iterrows():
        label = f"[{row['rule_id']}] {row['rule_name']} (Priority {row['priority']})"
        options[label] = int(row["rule_id"])

    with st.form("add_rule_version_form", clear_on_submit=False):
        selected_label = st.selectbox("Select Rule", options=list(options.keys()))
        version_number = st.text_input("Version Number", placeholder="e.g., 1.0, v2")
        rule_snapshot = st.text_area("Rule Snapshot (JSON text)", placeholder='e.g., {"name": "My Rule", "conditions": []}')
        is_active = st.checkbox("Is Active", value=False)
        submitted = st.form_submit_button("Add Rule Version")

        if submitted:
            if not version_number or not version_number.strip():
                st.warning("Please provide a version number.")
            else:
                rule_id = options[selected_label]
                try:
                    add_rule_version(rule_id, version_number.strip(), rule_snapshot.strip() if rule_snapshot else "{}", is_active)
                    st.success("Rule version added successfully.")
                    st.toast("Rule version added.", icon="✅")
                except Exception as exc:
                    st.error("Unable to add rule version. Please check the database and constraints.")
                    with st.expander("Error details"):
                        st.code(str(exc))


def show_add_rule_priority_page(user_id: int) -> None:
    st.subheader("Add Rule Priority")
    st.caption("Assign a priority level to a rule. Each rule may have only one priority entry.")

    try:
        rules_df = fetch_rules(user_id)
    except Exception as exc:
        st.error("Unable to load rules from the database.")
        with st.expander("Error details"):
            st.code(str(exc))
        return

    if rules_df.empty:
        st.info("No rules found. Please add a rule first.")
        return

    options: Dict[str, int] = {}
    for _, row in rules_df.iterrows():
        label = f"[{row['rule_id']}] {row['rule_name']} (Priority {row['priority']})"
        options[label] = int(row["rule_id"])

    with st.form("add_rule_priority_form", clear_on_submit=False):
        selected_label = st.selectbox("Select Rule", options=list(options.keys()))
        priority_level = st.number_input("Priority Level", min_value=0, step=1, value=1)
        submitted = st.form_submit_button("Add Rule Priority")

        if submitted:
            rule_id = options[selected_label]
            if rule_priority_exists(rule_id):
                st.error("This rule already has a priority. Only one priority per rule is allowed.")
            else:
                try:
                    add_rule_priority(rule_id, int(priority_level))
                    st.success("Rule priority added successfully.")
                    st.toast("Rule priority added.", icon="✅")
                    fetch_rules.clear()
                except Exception as exc:
                    st.error("Unable to add rule priority. Please check the database and constraints.")
                    with st.expander("Error details"):
                        st.code(str(exc))


def show_conflicts_page(user_id: int) -> None:
    st.subheader("Detected Rule Conflicts")
    st.caption(
        "This view reads directly from the SQL view `rule_conflicts_view`. "
        "All conflict detection logic is implemented in the database."
    )

    try:
        df = fetch_conflicts(user_id)
    except Exception as exc:
        st.error("Unable to fetch conflicts from `rule_conflicts_view`.")
        with st.expander("Error details"):
            st.code(str(exc))
        return

    if df.empty:
        st.info("No conflicts found in `rule_conflicts_view`.")
        return

    st.markdown(f"**Total Conflicts:** {len(df)}")
    st.dataframe(
    df,
    use_container_width=True,
    column_config={
        "conflict_reason": st.column_config.TextColumn(
            "Conflict Explanation",
            width="large",
        )
    }
)

    st.subheader("AI Conflict Explanation")
    st.caption(
        "Optional: get a plain-language explanation and resolution suggestions. "
        "The AI does not modify data, resolve conflicts, run SQL, or change priorities. "
        "If the AI request fails or is rate-limited, a deterministic fallback explanation is shown."
    )
    if not HUGGINGFACE_API_KEY:
        st.warning("Set `HUGGINGFACE_API_KEY` in your environment to enable AI explanations. A fallback explanation is always available.")

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        rule_1_name = _get_conflict_field(row_dict, "rule_1_name", "rule_name_1", default="Rule 1")
        rule_2_name = _get_conflict_field(row_dict, "rule_2_name", "rule_name_2", default="Rule 2")
        with st.expander(f"{rule_1_name} vs {rule_2_name}"):
            if st.session_state.get("ai_explain_results", {}).get(idx):
                st.markdown(st.session_state["ai_explain_results"][idx])
            if st.button("Explain with AI", key=f"ai_explain_{idx}"):
                st.session_state.setdefault("ai_explain_results", {})[idx] = ""
                with st.spinner("Getting AI explanation..."):
                    result = explain_conflict_ai(row_dict)
                    if result.strip().startswith("*") or "unavailable" in result:
                        result = explain_conflict_fallback(row_dict)
                    st.session_state["ai_explain_results"][idx] = result
                st.rerun()


def show_all_rules_overview_page(user_id: int) -> None:
    st.subheader("Rules")
    st.caption("All rules in the database.")
    
    try:
        rules_df = fetch_all_rules(user_id)
        if rules_df.empty:
            st.info("No rules found in the database.")
        else:
            st.dataframe(rules_df, use_container_width=True)
    except Exception as exc:
        st.error("Unable to fetch rules from the database.")
        with st.expander("Error details"):
            st.code(str(exc))
    
    st.subheader("Conditions")
    st.caption("All rule conditions joined with rule names.")
    
    try:
        conditions_df = fetch_all_conditions(user_id)
        if conditions_df.empty:
            st.info("No conditions found in the database.")
        else:
            st.dataframe(conditions_df, use_container_width=True)
    except Exception as exc:
        st.error("Unable to fetch conditions from the database.")
        with st.expander("Error details"):
            st.code(str(exc))
    
    st.subheader("Actions")
    st.caption("All rule actions joined with rule names.")
    
    try:
        actions_df = fetch_all_actions(user_id)
        if actions_df.empty:
            st.info("No actions found in the database.")
        else:
            st.dataframe(actions_df, use_container_width=True)
    except Exception as exc:
        st.error("Unable to fetch actions from the database.")
        with st.expander("Error details"):
            st.code(str(exc))


def show_manage_delete_rules_page(user_id: int) -> None:
    st.subheader("Delete Rule")
    st.caption("Delete a rule and all its related conditions and actions (CASCADE).")
    
    # Display all rules in a table
    try:
        rules_df = fetch_rules_for_delete(user_id)
        if rules_df.empty:
            st.info("No rules found in the database.")
            return
        
        st.dataframe(rules_df[['rule_id', 'rule_name']], use_container_width=True)
        
        # Build dropdown options
        rule_options: Dict[str, int] = {}
        for _, row in rules_df.iterrows():
            label = f"[{row['rule_id']}] {row['rule_name']}"
            rule_options[label] = int(row["rule_id"])
        
        with st.form("delete_rule_form", clear_on_submit=False):
            selected_label = st.selectbox("Select Rule to Delete", options=list(rule_options.keys()))
            confirm_delete = st.checkbox("I confirm that I want to delete this rule and all its related conditions and actions")
            submitted = st.form_submit_button("Delete Rule", type="primary")
            
            if submitted:
                if not confirm_delete:
                    st.warning("Please confirm deletion by checking the checkbox.")
                else:
                    rule_id = rule_options[selected_label]
                    try:
                        delete_rule(rule_id, user_id)
                        st.success(f"Rule {rule_id} deleted successfully. All related conditions and actions have been deleted.")
                        st.toast("Rule deleted.", icon="✅")
                        # Clear all caches to refresh UI
                        fetch_rules.clear()
                        fetch_all_rules.clear()
                        fetch_rules_for_delete.clear()
                        fetch_all_conditions.clear()
                        fetch_conditions_for_delete.clear()
                        fetch_all_actions.clear()
                        fetch_actions_for_delete.clear()
                        fetch_conflicts.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error("Unable to delete rule. Please check the database connection and constraints.")
                        with st.expander("Error details"):
                            st.code(str(exc))
    
    except Exception as exc:
        st.error("Unable to load rules from the database.")
        with st.expander("Error details"):
            st.code(str(exc))
        return
    
    st.markdown("---")
    
    # Delete Individual Conditions section
    with st.expander("Delete Individual Conditions", expanded=False):
        st.caption("Delete a specific condition without deleting the entire rule.")
        
        try:
            conditions_df = fetch_conditions_for_delete(user_id)
            if conditions_df.empty:
                st.info("No conditions found in the database.")
            else:
                st.dataframe(conditions_df, use_container_width=True)
                
                # Build dropdown options
                condition_options: Dict[str, int] = {}
                for _, row in conditions_df.iterrows():
                    label = f"[{row['condition_id']}] Rule: {row['rule_name']} | {row['attribute_name']} {row['operator']} {row['value']}"
                    condition_options[label] = int(row["condition_id"])
                
                with st.form("delete_condition_form", clear_on_submit=False):
                    selected_label = st.selectbox("Select Condition to Delete", options=list(condition_options.keys()))
                    confirm_delete = st.checkbox("I confirm that I want to delete this condition")
                    submitted = st.form_submit_button("Delete Condition", type="primary")
                    
                    if submitted:
                        if not confirm_delete:
                            st.warning("Please confirm deletion by checking the checkbox.")
                        else:
                            condition_id = condition_options[selected_label]
                            try:
                                delete_condition(condition_id)
                                st.success(f"Condition {condition_id} deleted successfully.")
                                st.toast("Condition deleted.", icon="✅")
                                # Clear caches to refresh UI
                                fetch_all_conditions.clear()
                                fetch_conditions_for_delete.clear()
                                fetch_conflicts.clear()
                                st.rerun()
                            except Exception as exc:
                                st.error("Unable to delete condition. Please check the database connection and constraints.")
                                with st.expander("Error details"):
                                    st.code(str(exc))
        
        except Exception as exc:
            st.error("Unable to load conditions from the database.")
            with st.expander("Error details"):
                st.code(str(exc))
    
    st.markdown("---")
    
    # Delete Individual Actions section
    with st.expander("Delete Individual Actions", expanded=False):
        st.caption("Delete a specific action without deleting the entire rule.")
        
        try:
            actions_df = fetch_actions_for_delete(user_id)
            if actions_df.empty:
                st.info("No actions found in the database.")
            else:
                st.dataframe(actions_df, use_container_width=True)
                
                # Build dropdown options
                action_options: Dict[str, int] = {}
                for _, row in actions_df.iterrows():
                    label = f"[{row['action_id']}] Rule: {row['rule_name']} | {row['action_type']} {row['target_entity']}"
                    action_options[label] = int(row["action_id"])
                
                with st.form("delete_action_form", clear_on_submit=False):
                    selected_label = st.selectbox("Select Action to Delete", options=list(action_options.keys()))
                    confirm_delete = st.checkbox("I confirm that I want to delete this action")
                    submitted = st.form_submit_button("Delete Action", type="primary")
                    
                    if submitted:
                        if not confirm_delete:
                            st.warning("Please confirm deletion by checking the checkbox.")
                        else:
                            action_id = action_options[selected_label]
                            try:
                                delete_action(action_id)
                                st.success(f"Action {action_id} deleted successfully.")
                                st.toast("Action deleted.", icon="✅")
                                # Clear caches to refresh UI
                                fetch_all_actions.clear()
                                fetch_actions_for_delete.clear()
                                fetch_conflicts.clear()
                                st.rerun()
                            except Exception as exc:
                                st.error("Unable to delete action. Please check the database connection and constraints.")
                                with st.expander("Error details"):
                                    st.code(str(exc))
        
        except Exception as exc:
            st.error("Unable to load actions from the database.")
            with st.expander("Error details"):
                st.code(str(exc))


def show_schema_data_overview_page() -> None:
    """Read-only overview of entire database schema and data (users table excluded)."""
    st.subheader("Rules")
    try:
        df = fetch_dataframe("SELECT * FROM rules")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Unable to fetch rules.")
        with st.expander("Error details"):
            st.code(str(exc))

    st.subheader("Rule Conditions")
    try:
        df = fetch_dataframe("SELECT * FROM rule_conditions")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Unable to fetch rule_conditions.")
        with st.expander("Error details"):
            st.code(str(exc))

    st.subheader("Rule Actions")
    try:
        df = fetch_dataframe("SELECT * FROM rule_actions")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Unable to fetch rule_actions.")
        with st.expander("Error details"):
            st.code(str(exc))

    st.subheader("Rule Conflicts")
    try:
        df = fetch_dataframe("SELECT * FROM rule_conflicts_view")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "conflict_reason": st.column_config.TextColumn(
                    "Conflict Reason",
                    width="large",
                )
            }
        )
    except Exception as exc:
        st.error("Unable to fetch rule_conflicts_view.")
        with st.expander("Error details"):
            st.code(str(exc))

    st.subheader("Rule Status")
    try:
        df = fetch_dataframe("SELECT * FROM rule_status")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Unable to fetch rule_status.")
        with st.expander("Error details"):
            st.code(str(exc))

    st.subheader("Rule Versions")
    try:
        df = fetch_dataframe("SELECT * FROM rule_versions")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Unable to fetch rule_versions.")
        with st.expander("Error details"):
            st.code(str(exc))

    st.subheader("Rule Priority")
    try:
        df = fetch_dataframe("SELECT * FROM rule_priority")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Unable to fetch rule_priority.")
        with st.expander("Error details"):
            st.code(str(exc))

    st.subheader("Audit Log")
    try:
        df = fetch_dataframe("SELECT * FROM audit_log")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Unable to fetch audit_log.")
        with st.expander("Error details"):
            st.code(str(exc))

    st.subheader("Rule Execution Log")
    try:
        df = fetch_dataframe("SELECT * FROM rule_execution_log")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("Unable to fetch rule_execution_log.")
        with st.expander("Error details"):
            st.code(str(exc))


def main() -> None:

    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.user_id is None:
        st.title("Rule Conflict Database")
        login_page()
        return

    st.title("Rule Conflict Database")
    show_sidebar(st.session_state.user_id, st.session_state.username)
    user_id = st.session_state.user_id

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "All Rules Overview",
        "Add Rule Status",
        "Add Rule",
        "Add Rule Condition",
        "Add Rule Action",
        "Add Rule Version",
        "Add Rule Priority",
        "View Conflicts",
        "Manage / Delete Rules",
        "Schema & Data Overview"
    ])

    with tab1:
        show_all_rules_overview_page(user_id)

    with tab2:
        show_add_rule_status_page()

    with tab3:
        show_add_rule_page(user_id)

    with tab4:
        show_add_condition_page(user_id)

    with tab5:
        show_add_action_page(user_id)

    with tab6:
        show_add_rule_version_page(user_id)

    with tab7:
        show_add_rule_priority_page(user_id)

    with tab8:
        show_conflicts_page(user_id)

    with tab9:
        show_manage_delete_rules_page(user_id)

    with tab10:
        show_schema_data_overview_page()


if __name__ == "__main__":
    main()