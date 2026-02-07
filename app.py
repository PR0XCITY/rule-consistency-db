import streamlit as st
import psycopg2
import pandas as pd
from datetime import date
from typing import Any, Dict, Optional, Tuple
import os

from huggingface_hub import InferenceClient
from groq import Groq

import streamlit as st

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client: Optional[Groq] = None
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
except Exception:
    groq_client = None

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


# Hugging Face: official InferenceClient (no manual HTTP / deprecated URLs)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
_HF_CLIENT: Optional[InferenceClient] = None


def _get_hf_client() -> Optional[InferenceClient]:
    """Lazy-init InferenceClient with model and token from env."""
    global _HF_CLIENT
    if _HF_CLIENT is None and HUGGINGFACE_API_KEY:
        try:
            _HF_CLIENT = InferenceClient(
                model="google/flan-t5-base",
                token=HUGGINGFACE_API_KEY,
            )
        except Exception:
            _HF_CLIENT = False  # type: ignore
    return _HF_CLIENT if _HF_CLIENT else None

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


def get_rule_conditions_summary(rule_id: int) -> str:
    """
    Query rule_conditions for the given rule_id, order by condition_id, and return
    a readable string, e.g. "gpa >= 8 AND year = 1". Returns "No conditions defined" if none exist.
    """
    try:
        query = """
            SELECT rc.attribute_name, rc.operator, rc.value
            FROM rule_conditions rc
            WHERE rc.rule_id = %s
            ORDER BY rc.condition_id
        """
        df = fetch_dataframe(query, (int(rule_id),))
        if df.empty:
            return "No conditions defined"
        parts = []
        for _, row in df.iterrows():
            attr = str(row.get("attribute_name", "")).strip()
            op = str(row.get("operator", "")).strip()
            val = str(row.get("value", "")).strip()
            if attr or op or val:
                parts.append(f"{attr} {op} {val}")
        return " AND ".join(parts) if parts else "No conditions defined"
    except Exception:
        return "No conditions defined"


def explain_conflict_deterministic(conflict: dict) -> str:
    """
    Local, deterministic explanation based on rule conditions and actions. No generic templates;
    output is specific to the given condition strings and conflict data.
    """
    rule_1_name = _get_conflict_field(conflict, "rule_1_name", "rule_name_1", default="Rule 1")
    rule_2_name = _get_conflict_field(conflict, "rule_2_name", "rule_name_2", default="Rule 2")
    rule_1_conditions = _get_conflict_field(conflict, "rule_1_conditions", default="No conditions defined")
    rule_2_conditions = _get_conflict_field(conflict, "rule_2_conditions", default="No conditions defined")
    action_1 = _get_conflict_field(conflict, "action_1", "action_type_1", default="action 1")
    action_2 = _get_conflict_field(conflict, "action_2", "action_type_2", default="action 2")
    target_entity = _get_conflict_field(conflict, "target_entity", "target", default="target")

    no_c1 = not rule_1_conditions or rule_1_conditions == "No conditions defined"
    no_c2 = not rule_2_conditions or rule_2_conditions == "No conditions defined"
    if no_c1 and no_c2:
        overlap = "Both rules have no conditions defined, so they apply to the same set of entities."
    elif no_c1:
        overlap = f"Rule 1 has no conditions (applies to all); Rule 2 applies when: {rule_2_conditions}. So any entity satisfying {rule_2_conditions} can trigger both rules."
    elif no_c2:
        overlap = f"Rule 2 has no conditions (applies to all); Rule 1 applies when: {rule_1_conditions}. So any entity satisfying {rule_1_conditions} can trigger both rules."
    else:
        overlap = f"Rule 1 applies when: {rule_1_conditions}. Rule 2 applies when: {rule_2_conditions}. If an entity satisfies both ({rule_1_conditions} AND {rule_2_conditions}), both rules fire for the same {target_entity}."

    action_conflict = (
        f"Rule \"{rule_1_name}\" specifies action **{action_1}** and rule \"{rule_2_name}\" specifies action **{action_2}**. "
        f"For a single {target_entity} that meets both condition sets, the system cannot apply {action_1} and {action_2} at once, so this is a conflict."
    )

    strategies = (
        "**Resolution strategies:** (1) Assign different priorities to the two rules so one is applied first for overlapping cases. "
        "(2) Tighten one or both condition sets so they do not overlap (e.g. add or refine attributes). "
        "(3) Merge or remove one rule if the intended outcome can be achieved by the other."
    )

    return f"**Condition overlap**\n\n{overlap}\n\n**Why the actions conflict**\n\n{action_conflict}\n\n{strategies}"


def explain_conflict_groq(conflict: dict) -> Optional[str]:
    """
    Use Groq API for conflict explanation. Returns generated text on success, None on any failure.
    """
    rule_1_name = _get_conflict_field(conflict, "rule_1_name", "rule_name_1", default="Rule 1")
    rule_2_name = _get_conflict_field(conflict, "rule_2_name", "rule_name_2", default="Rule 2")
    rule_1_conditions = _get_conflict_field(conflict, "rule_1_conditions", default="No conditions defined")
    rule_2_conditions = _get_conflict_field(conflict, "rule_2_conditions", default="No conditions defined")
    action_1 = _get_conflict_field(conflict, "action_1", "action_type_1", default="action 1")
    action_2 = _get_conflict_field(conflict, "action_2", "action_type_2", default="action 2")
    target_entity = _get_conflict_field(conflict, "target_entity", "target", default="target")

    prompt = (
        "Rule conflict analysis.\n\n"
        f"Rule 1: Name={rule_1_name}, Conditions={rule_1_conditions}, Action={action_1}.\n"
        f"Rule 2: Name={rule_2_name}, Conditions={rule_2_conditions}, Action={action_2}.\n"
        f"Target entity: {target_entity}.\n\n"
        "1) Determine if both rules can apply simultaneously. "
        "2) Explain why this is a conflict or not. "
        "3) Suggest two resolution strategies."
    )

    if not groq_client:
        return None
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.4,
        )
        if not response or not response.choices:
            return None
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception:
        return None


def explain_conflict_ai(conflict: dict) -> Optional[str]:
    """
    Use Hugging Face InferenceClient for a conflict-specific explanation. On any failure, returns None.
    """
    rule_1_name = _get_conflict_field(conflict, "rule_1_name", "rule_name_1", default="Rule 1")
    rule_2_name = _get_conflict_field(conflict, "rule_2_name", "rule_name_2", default="Rule 2")
    rule_1_conditions = _get_conflict_field(conflict, "rule_1_conditions", default="No conditions defined")
    rule_2_conditions = _get_conflict_field(conflict, "rule_2_conditions", default="No conditions defined")
    action_1 = _get_conflict_field(conflict, "action_1", "action_type_1", default="action 1")
    action_2 = _get_conflict_field(conflict, "action_2", "action_type_2", default="action 2")
    target_entity = _get_conflict_field(conflict, "target_entity", "target", default="target")

    prompt = (
        "Rule conflict analysis.\n\n"
        f"Rule 1: Name={rule_1_name}, Conditions={rule_1_conditions}, Action={action_1}.\n"
        f"Rule 2: Name={rule_2_name}, Conditions={rule_2_conditions}, Action={action_2}.\n"
        f"Target entity: {target_entity}.\n\n"
        "1) Determine whether both rules can apply to the same real-world case. "
        "2) Explain why this is or is not a conflict. "
        "3) Suggest at least two resolution strategies."
    )

    if not HUGGINGFACE_API_KEY:
        return None
    client = _get_hf_client()
    if client is None:
        return None

    try:
        result = client.text_generation(
            prompt,
            max_new_tokens=256,
            temperature=0.4,
        )
    except Exception:
        return None

    if result is None:
        return None
    text = (result if isinstance(result, str) else getattr(result, "generated_text", None) or str(result)).strip()
    if not text:
        return None
    return text


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
        "Get a plain-language explanation from the AI. Explanations are based on rule conditions and actions. "
        "The AI does not modify data, resolve conflicts, run SQL, or change priorities. "
        "AI explanations are optional and may be unavailable."
    )

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        rule_1_name = _get_conflict_field(row_dict, "rule_1_name", "rule_name_1", default="Rule 1")
        rule_2_name = _get_conflict_field(row_dict, "rule_2_name", "rule_name_2", default="Rule 2")
        with st.expander(f"{rule_1_name} vs {rule_2_name}"):
            if st.session_state.get("ai_explain_results", {}).get(idx):
                if st.session_state.get("ai_explain_used_deterministic", {}).get(idx):
                    st.caption("AI service unavailable, showing system-generated explanation.")
                st.markdown(st.session_state["ai_explain_results"][idx])
            if st.button("Explain with AI", key=f"ai_explain_{idx}"):
                st.session_state.setdefault("ai_explain_results", {})[idx] = ""
                with st.spinner("Getting AI explanation..."):
                    enriched = dict(row_dict)
                    rid1 = row_dict.get("rule_1_id") or row_dict.get("rule_id_1")
                    rid2 = row_dict.get("rule_2_id") or row_dict.get("rule_id_2")
                    try:
                        if rid1 is not None and (not hasattr(pd, "isna") or not pd.isna(rid1)):
                            enriched["rule_1_conditions"] = get_rule_conditions_summary(int(rid1))
                        else:
                            enriched["rule_1_conditions"] = "No conditions defined"
                        if rid2 is not None and (not hasattr(pd, "isna") or not pd.isna(rid2)):
                            enriched["rule_2_conditions"] = get_rule_conditions_summary(int(rid2))
                        else:
                            enriched["rule_2_conditions"] = "No conditions defined"
                    except (TypeError, ValueError):
                        enriched["rule_1_conditions"] = "No conditions defined"
                        enriched["rule_2_conditions"] = "No conditions defined"

                    ai_text = explain_conflict_groq(enriched)
                    if ai_text:
                        st.session_state["ai_explain_results"][idx] = "[AI Explanation]\n\n" + ai_text
                        st.session_state.setdefault("ai_explain_used_deterministic", {})[idx] = False
                    else:
                        st.session_state.setdefault("ai_explain_used_deterministic", {})[idx] = True
                        det_text = explain_conflict_deterministic(enriched)
                        st.session_state["ai_explain_results"][idx] = "[System Explanation]\n\n" + det_text
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