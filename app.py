import streamlit as st
import psycopg2
import pandas as pd
from datetime import date
from typing import Any, Dict, Optional, Tuple
import os

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



def add_rule(rule_name: str, category: str, priority: int, active_from: date, active_to: date) -> None:
    query = """
        INSERT INTO rules (rule_name, rule_category, priority, active_from, active_to)
        VALUES (%s, %s, %s, %s, %s)
    """
    params = (rule_name, category, priority, active_from, active_to)
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


@st.cache_data(ttl=60)
def fetch_rules() -> pd.DataFrame:
    """Return available rules for selection (cached briefly for smoother UX)."""
    query = "SELECT rule_id, rule_name, rule_category, priority FROM rules ORDER BY rule_id"
    return fetch_dataframe(query)


@st.cache_data(ttl=60)
def fetch_conflicts() -> pd.DataFrame:

    query = "SELECT * FROM rule_conflicts_view"
    return fetch_dataframe(query)


@st.cache_data(ttl=60)
def fetch_all_rules() -> pd.DataFrame:
    """Return all rules for the overview tab."""
    query = "SELECT * FROM rules ORDER BY rule_id"
    return fetch_dataframe(query)


@st.cache_data(ttl=60)
def fetch_all_conditions() -> pd.DataFrame:
    """Return all rule conditions joined with rule names."""
    query = """
        SELECT 
            rc.condition_id,
            rc.rule_id,
            r.rule_name,
            rc.attribute_name,
            rc.operator,
            rc.value
        FROM rule_conditions rc
        JOIN rules r ON rc.rule_id = r.rule_id
        ORDER BY rc.rule_id, rc.condition_id
    """
    return fetch_dataframe(query)


@st.cache_data(ttl=60)
def fetch_all_actions() -> pd.DataFrame:
    """Return all rule actions joined with rule names."""
    query = """
        SELECT 
            ra.action_id,
            ra.rule_id,
            r.rule_name,
            ra.action_type,
            ra.target_entity
        FROM rule_actions ra
        JOIN rules r ON ra.rule_id = r.rule_id
        ORDER BY ra.rule_id, ra.action_id
    """
    return fetch_dataframe(query)


def check_db_health() -> Tuple[bool, Optional[str]]:

    try:

        _ = fetch_dataframe("SELECT 1 AS ok")
        return True, None
    except Exception as exc:
        return False, str(exc)



def show_sidebar() -> None:

    st.sidebar.title("Rule Conflict DB")
    st.sidebar.caption("Manage rules, conditions, actions, and view conflicts.")

    # Database status
    healthy, error_msg = check_db_health()
    if healthy:
        st.sidebar.success("Database connected", icon="✅")
    else:
        st.sidebar.error("Database error", icon="⚠️")
        with st.sidebar.expander("Details", expanded=False):
            st.code(error_msg or "Unknown error")

    st.sidebar.markdown("---")
    st.sidebar.caption("All rule evaluation and conflict detection is implemented in SQL.")


def show_add_rule_page() -> None:
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
                    add_rule(rule_name.strip(), rule_category.strip(), int(priority), active_from, active_to)
                    st.success("Rule added successfully.")
                    st.toast("Rule added.", icon="✅")
                    # Clear rule cache so new rule appears in dropdowns and overview
                    fetch_rules.clear()
                    fetch_all_rules.clear()
                except Exception as exc:
                    st.error("Unable to add rule. Please check the database connection and constraints.")
                    with st.expander("Error details"):
                        st.code(str(exc))


def show_add_condition_page() -> None:
    st.subheader("Add Condition")
    st.caption("Attach a condition to an existing rule.")

    try:
        rules_df = fetch_rules()
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
                except Exception as exc:
                    st.error("Unable to add condition. Please check the database and constraints.")
                    with st.expander("Error details"):
                        st.code(str(exc))


def show_add_action_page() -> None:
    st.subheader("Add Action")
    st.caption("Attach an action to an existing rule.")

    try:
        rules_df = fetch_rules()
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
                except Exception as exc:
                    st.error("Unable to add action. Please check the database and constraints.")
                    with st.expander("Error details"):
                        st.code(str(exc))


def show_conflicts_page() -> None:
    st.subheader("Detected Rule Conflicts")
    st.caption(
        "This view reads directly from the SQL view `rule_conflicts_view`. "
        "All conflict detection logic is implemented in the database."
    )

    try:
        df = fetch_conflicts()
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


def show_all_rules_overview_page() -> None:
    st.subheader("Rules")
    st.caption("All rules in the database.")
    
    try:
        rules_df = fetch_all_rules()
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
        conditions_df = fetch_all_conditions()
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
        actions_df = fetch_all_actions()
        if actions_df.empty:
            st.info("No actions found in the database.")
        else:
            st.dataframe(actions_df, use_container_width=True)
    except Exception as exc:
        st.error("Unable to fetch actions from the database.")
        with st.expander("Error details"):
            st.code(str(exc))


def main() -> None:

    st.title("Rule Conflict Database")

    show_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "All Rules Overview",
        "Add Rule",
        "Add Condition",
        "Add Action",
        "View Conflicts"
    ])

    with tab1:
        show_all_rules_overview_page()

    with tab2:
        show_add_rule_page()

    with tab3:
        show_add_condition_page()

    with tab4:
        show_add_action_page()

    with tab5:
        show_conflicts_page()


if __name__ == "__main__":
    main()
