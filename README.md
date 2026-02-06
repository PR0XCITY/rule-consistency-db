# Rule Conflict Database

## Overview
The Rule Conflict Database is a DBMS-focused project that explores how logical conflicts between rules can be modeled and detected using SQL. Instead of managing traditional entities (students, orders, etc.), this project treats **rules themselves as data** and uses relational queries to identify inconsistencies between them.

The project intentionally focuses on **conflict detection**, not automatic resolution, to highlight both the capabilities and limitations of relational databases in reasoning about rule-based systems.

---

## Problem Statement
In real-world systems, rules are often added incrementally by different stakeholders. Over time, these rules may overlap, contradict, or interfere with each other. Traditional database systems store rules but do not actively detect logical conflicts between them.

This project aims to:
- Model rules in a structured relational format
- Detect conflicts between rules using SQL
- Demonstrate the challenges of expressing logical reasoning in a DBMS

---

## Key Idea
A **rule conflict** is identified when:
- Two rules apply to the same target entity
- Their active time periods overlap
- Their actions contradict each other (e.g., `allow` vs `deny`)

All conflict detection logic is implemented **inside the database** using SQL views.  
Python is used only for data input and visualization.

---

## Technologies Used
- PostgreSQL
- SQL
- Python
- Streamlit
- psycopg2

---

## Database Schema

### Tables
- `rules`  
  Stores rule metadata such as name, category, priority, and active period.

- `rule_conditions`  
  Stores the conditions under which a rule applies.

- `rule_actions`  
  Stores the action performed by a rule on a target entity.

### View
- `rule_conflicts_view`  
  Detects conflicting rules based on overlapping time periods and contradictory actions.

---

## Current Implementation Scope (Intentional)

### Implemented
- Normalized relational schema for rule storage
- SQL-based detection of direct action-level conflicts
- Clear separation between database logic and application layer
- UI for inserting rules, conditions, actions, and viewing detected conflicts

### Not Implemented (By Design)
- Logical evaluation of overlapping conditions
- Multi-condition rule reasoning
- Priority-based conflict resolution
- Automatic conflict resolution

These are intentionally left as future scope to emphasize DBMS limitations.

---

## Why This Project Is Different
Most DBMS projects focus on managing entities. This project focuses on **managing logic**.

It demonstrates:
- How SQL can be stretched to reason about rule consistency
- Where relational databases fall short for logical inference
- Why some problems require tradeoffs instead of complete automation

---

## Project Status
This project represents a **partial but deliberate implementation**.  
The focus is on problem modeling and conflict detection rather than feature completeness.

This approach reflects real-world database design, where understanding limitations is as important as implementation.

---

## How to Run the Project

### 1. Prerequisites
- PostgreSQL installed and running
- Python 3.9 or higher
- `pip` available

---

### 2. Create Database
Create a PostgreSQL database named:

```sql
CREATE DATABASE rule_conflict_db;
