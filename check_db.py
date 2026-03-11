import sqlite3
import os

db_path = "mlflow.db"
if os.path.exists(db_path):
    print(f"Checking {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT run_uuid, status FROM runs WHERE status = 'RUNNING'")
        running_runs = cursor.fetchall()
        if running_runs:
            print("Found active runs in local DB:")
            for run_id, status in running_runs:
                print(f"  - Run ID: {run_id}, Status: {status}")
        else:
            print("No active runs found in local DB.")
    except Exception as e:
        print(f"Error checking DB: {e}")
    finally:
        conn.close()
else:
    print("mlflow.db not found.")
