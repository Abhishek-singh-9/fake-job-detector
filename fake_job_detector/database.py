"""
Database Access Layer for Fake Job Posting Detector.

Manages all persistent storage using Python's standard library `sqlite3`:
1. User accounts and role-based permissions (user/admin).
2. Scan history audit records (predictions, risk ratings, rule warnings, feature flags).
3. Summary analytics for dashboard statistics.

Security & Architectural Principles (Essential for Viva):
- Parameterized Queries Everywhere:
  Every query uses parameter binding (`?` placeholders). User input is never concatenated
  directly into SQL strings, completely preventing SQL Injection attacks.
- Foreign Key Constraints & Cascading:
  Enforces `PRAGMA foreign_keys = ON;` on every database connection. When a user is deleted,
  all associated scan history rows in `checks` are automatically deleted via `ON DELETE CASCADE`.
- Ownership Authorization in Delete Queries:
  Deletions require both `check_id` AND `user_id` (`DELETE FROM checks WHERE id = ? AND user_id = ?`).
  This guarantees that User A cannot delete or tamper with User B's scan history.
- Structured Serialization:
  Complex nested structures (`warnings` list, `rule_flags` dict) are serialized and
  deserialized cleanly using JSON.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

# Default database location inside the project's data directory
DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent / "data" / "app.db"


class Database:
    """
    SQLite database manager providing thread-safe connections and helper queries.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initializes database connection settings and ensures schema is created.

        Args:
            db_path (Optional[Path]): Path to SQLite database file. Defaults to data/app.db.
        """
        self.db_path: Path = db_path if db_path is not None else DEFAULT_DB_PATH
        # Ensure parent directory (e.g. data/) exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Create database tables if they do not already exist
        self._create_tables()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for robust database connections.

        Why this matters for viva:
        - Automatically commits on successful execution.
        - Automatically rolls back transactions if an exception occurs.
        - Always closes the connection in the finally block, preventing database locks.
        - Enforces foreign key constraints on every opened connection.

        Yields:
            sqlite3.Connection: Configured SQLite database connection.
        """
        conn = sqlite3.connect(self.db_path)
        # sqlite3.Row allows column access by name as well as index
        conn.row_factory = sqlite3.Row
        # Foreign key enforcement must be explicitly enabled in SQLite per connection
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self) -> None:
        """
        Initializes relational schema for users and scan checks.
        """
        with self._get_connection() as conn:
            # Users table: stores authentication details and role
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
                    created_at TEXT NOT NULL
                );
            """)

            # Checks table: stores historical job scans linked to a user
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    created_at TEXT NOT NULL,
                    job_title TEXT,
                    job_text TEXT,
                    fraud_probability REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    warnings TEXT NOT NULL,
                    rule_flags TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                );
            """)

    # -------------------------------------------------------------------------
    # User Management Methods
    # -------------------------------------------------------------------------

    def add_user(self, username: str, password_hash: str, role: str = "user") -> int:
        """
        Inserts a new user into the database.

        Args:
            username (str): Unique username.
            password_hash (str): Bcrypt-hashed password.
            role (str): Role designation ('user' or 'admin').

        Returns:
            int: The inserted user's ID.

        Raises:
            ValueError: If the username already exists.
        """
        now: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    (username, password_hash, role, now),
                )
                return int(cursor.lastrowid)
            except sqlite3.IntegrityError as e:
                # Catch UNIQUE constraint failure on username
                raise ValueError(f"Username '{username}' already exists.") from e

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a user record by username.

        Args:
            username (str): Username to query.

        Returns:
            Optional[Dict[str, Any]]: User record dictionary including password_hash, or None.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, username, password_hash, role, created_at
                FROM users
                WHERE username = ?;
                """,
                (username,),
            )
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def count_admins(self) -> int:
        """
        Returns the total number of users with the 'admin' role.

        Returns:
            int: Number of administrator accounts.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin';")
            return int(cursor.fetchone()[0])

    def list_users(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all registered users along with their scan check count.

        Security Note for Viva:
        NEVER includes password_hash in the returned dictionaries to prevent accidental
        exposure in logs, UI tables, or JSON serialization.

        Returns:
            List[Dict[str, Any]]: List of user dictionaries containing 'id', 'username',
                                  'role', 'created_at', and 'check_count'.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT u.id, u.username, u.role, u.created_at,
                       COUNT(c.id) AS check_count
                FROM users u
                LEFT JOIN checks c ON u.id = c.user_id
                GROUP BY u.id
                ORDER BY u.id ASC;
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_user(self, user_id: int) -> bool:
        """
        Deletes a user and cascades deletion to their scan history.

        Rules:
        - The last remaining admin account cannot be deleted.

        Args:
            user_id (int): User ID to delete.

        Returns:
            bool: True if a user was deleted, False otherwise.

        Raises:
            ValueError: If attempting to delete the last remaining admin account.
        """
        with self._get_connection() as conn:
            # Check if user being deleted is an admin
            cursor = conn.execute("SELECT role FROM users WHERE id = ?;", (user_id,))
            row = cursor.fetchone()
            if row is not None and row["role"] == "admin":
                if self.count_admins() <= 1:
                    raise ValueError("Cannot delete the last remaining administrator account.")

            cursor = conn.execute(
                "DELETE FROM users WHERE id = ?;",
                (user_id,),
            )
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Scan History Methods
    # -------------------------------------------------------------------------

    def save_check(
        self,
        user_id: Optional[int],
        job_title: str,
        job_text: str,
        fraud_probability: float,
        risk_level: str,
        model_name: str,
        warnings: Any,
        rule_flags: Any,
    ) -> int:
        """
        Saves a job posting evaluation scan record.

        Args:
            user_id (Optional[int]): ID of the user performing the check (or None for guest).
            job_title (str): Title of the job posting.
            job_text (str): Job description text.
            fraud_probability (float): Computed fraud score (0.0 to 1.0).
            risk_level (str): 'low', 'medium', or 'high'.
            model_name (str): Architecture name of model used.
            warnings (Any): List of rule warning dicts (serialized to JSON).
            rule_flags (Any): Feature flags dict (serialized to JSON).

        Returns:
            int: The inserted check record ID.
        """
        now: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        warnings_json: str = json.dumps(warnings) if not isinstance(warnings, str) else warnings
        rule_flags_json: str = json.dumps(rule_flags) if not isinstance(rule_flags, str) else rule_flags

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO checks (
                    user_id, created_at, job_title, job_text,
                    fraud_probability, risk_level, model_name,
                    warnings, rule_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    user_id,
                    now,
                    job_title,
                    job_text,
                    float(fraud_probability),
                    risk_level.lower(),
                    model_name,
                    warnings_json,
                    rule_flags_json,
                ),
            )
            return int(cursor.lastrowid)

    def get_checks_for_user(
        self,
        user_id: int,
        risk_level: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves scan history for a specific user, sorted newest first.

        Supports optional filtering by risk level and job title substring search.

        Args:
            user_id (int): User ID.
            risk_level (Optional[str]): 'low', 'medium', or 'high'.
            search (Optional[str]): Substring search for job title.

        Returns:
            List[Dict[str, Any]]: List of check records with deserialized JSON fields.
        """
        query = """
            SELECT id, user_id, created_at, job_title, job_text,
                   fraud_probability, risk_level, model_name,
                   warnings, rule_flags
            FROM checks
            WHERE user_id = ?
        """
        params: List[Any] = [user_id]

        if risk_level:
            query += " AND LOWER(risk_level) = ?"
            params.append(risk_level.lower())

        if search:
            query += " AND LOWER(job_title) LIKE ?"
            params.append(f"%{search.lower()}%")

        query += " ORDER BY id DESC;"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            records = []
            for row in cursor.fetchall():
                rec = dict(row)
                rec["warnings"] = json.loads(rec["warnings"])
                rec["rule_flags"] = json.loads(rec["rule_flags"])
                records.append(rec)
            return records

    def get_check(self, check_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single scan check by its ID.

        Args:
            check_id (int): Check ID.

        Returns:
            Optional[Dict[str, Any]]: Check dictionary with parsed JSON fields, or None.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, user_id, created_at, job_title, job_text,
                       fraud_probability, risk_level, model_name,
                       warnings, rule_flags
                FROM checks
                WHERE id = ?;
                """,
                (check_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            rec = dict(row)
            rec["warnings"] = json.loads(rec["warnings"])
            rec["rule_flags"] = json.loads(rec["rule_flags"])
            return rec

    def get_check_with_owner(self, check_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single scan check by its ID, joined with the users table to
        include the username of the check's owner.

        Args:
            check_id (int): Scan check ID.

        Returns:
            Optional[Dict[str, Any]]: Check dictionary with parsed JSON fields and 'username', or None.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT c.id, c.user_id, u.username, c.created_at, c.job_title,
                       c.job_text, c.fraud_probability, c.risk_level, c.model_name,
                       c.warnings, c.rule_flags
                FROM checks c
                LEFT JOIN users u ON c.user_id = u.id
                WHERE c.id = ?;
                """,
                (check_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            rec = dict(row)
            rec["username"] = rec["username"] if rec["username"] is not None else "Guest"
            rec["warnings"] = json.loads(rec["warnings"])
            rec["rule_flags"] = json.loads(rec["rule_flags"])
            return rec

    def delete_check(self, check_id: int, user_id: Optional[int] = None) -> bool:
        """
        Deletes a check record.

        - If user_id is provided: Enforces ownership authorization (IDOR defense),
          deleting ONLY if the check belongs to that user.
        - If user_id is None: Allows administrative deletion of any check by ID.

        Args:
            check_id (int): Check record ID.
            user_id (Optional[int]): User ID asserting ownership, or None for admin.

        Returns:
            bool: True if deleted, False if record did not exist or belonged to another user.
        """
        with self._get_connection() as conn:
            if user_id is not None:
                cursor = conn.execute(
                    "DELETE FROM checks WHERE id = ? AND user_id = ?;",
                    (check_id, user_id),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM checks WHERE id = ?;",
                    (check_id,),
                )
            return cursor.rowcount > 0

    def delete_check_by_admin(self, check_id: int) -> bool:
        """
        Deletes a check record as an administrator by ID without ownership constraint.

        Args:
            check_id (int): Check record ID.

        Returns:
            bool: True if deleted, False otherwise.
        """
        return self.delete_check(check_id=check_id, user_id=None)

    def delete_all_checks_for_user(self, user_id: int) -> int:
        """
        Deletes all historical checks belonging to a user.

        Args:
            user_id (int): User ID.

        Returns:
            int: Number of deleted records.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM checks WHERE user_id = ?;",
                (user_id,),
            )
            return cursor.rowcount

    def get_all_checks(
        self,
        limit: int = 200,
        risk_level: Optional[str] = None,
        username: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves global recent checks across all users for administrator review,
        joined with the users table to include each check owner's username.

        Supports optional filtering by risk level and username.

        Args:
            limit (int): Maximum records to return (defaults to 200).
            risk_level (Optional[str]): Optional risk level filter ('low', 'medium', 'high').
            username (Optional[str]): Optional username filter.

        Returns:
            List[Dict[str, Any]]: List of recent checks with username joined.
        """
        query = """
            SELECT c.id, c.user_id, u.username, c.created_at, c.job_title,
                   c.job_text, c.fraud_probability, c.risk_level, c.model_name,
                   c.warnings, c.rule_flags
            FROM checks c
            LEFT JOIN users u ON c.user_id = u.id
            WHERE 1=1
        """
        params: List[Any] = []

        if risk_level and risk_level.lower() != "all":
            query += " AND LOWER(c.risk_level) = ?"
            params.append(risk_level.lower())

        if username and username.lower() != "all":
            query += " AND LOWER(u.username) = ?"
            params.append(username.lower())

        query += " ORDER BY c.id DESC LIMIT ?;"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            records = []
            for row in cursor.fetchall():
                rec = dict(row)
                rec["username"] = rec["username"] if rec["username"] is not None else "Guest"
                rec["warnings"] = json.loads(rec["warnings"])
                rec["rule_flags"] = json.loads(rec["rule_flags"])
                records.append(rec)
            return records

    # -------------------------------------------------------------------------
    # Analytics & Summary Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Computes aggregate usage and fraud risk statistics.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_checks: Total scans performed
                - total_users: Total registered users
                - high_risk_count: Total high-risk postings flagged
                - high_risk_percentage: Percentage of scans flagged high risk
                - checks_per_day: Daily scan counts for the past 14 days
        """
        with self._get_connection() as conn:
            # 1. Total checks
            cursor = conn.execute("SELECT COUNT(*) FROM checks;")
            total_checks: int = int(cursor.fetchone()[0])

            # 2. Total registered users
            cursor = conn.execute("SELECT COUNT(*) FROM users;")
            total_users: int = int(cursor.fetchone()[0])

            # 3. High risk count
            cursor = conn.execute("SELECT COUNT(*) FROM checks WHERE LOWER(risk_level) = 'high';")
            high_risk_count: int = int(cursor.fetchone()[0])

            # 4. High risk percentage
            high_risk_percentage: float = (
                round((high_risk_count / total_checks) * 100, 2)
                if total_checks > 0
                else 0.0
            )

            # 5. Daily checks for the last 14 days
            # Generate continuous 14-day date skeleton (ensures zero-count days are displayed)
            today = datetime.now().date()
            daily_skeleton: Dict[str, int] = {
                (today - timedelta(days=i)).strftime("%Y-%m-%d"): 0
                for i in range(13, -1, -1)
            }

            cursor = conn.execute("""
                SELECT substr(created_at, 1, 10) AS check_date, COUNT(*) AS count
                FROM checks
                WHERE created_at >= date('now', '-14 days')
                GROUP BY check_date
                ORDER BY check_date ASC;
            """)

            for row in cursor.fetchall():
                d_str = row["check_date"]
                if d_str in daily_skeleton:
                    daily_skeleton[d_str] = int(row["count"])

        return {
            "total_checks": total_checks,
            "total_users": total_users,
            "high_risk_count": high_risk_count,
            "high_risk_percentage": high_risk_percentage,
            "checks_per_day": daily_skeleton,
        }


# -----------------------------------------------------------------------------
# Self-Test Suite (Executed with: python database.py)
# -----------------------------------------------------------------------------

def run_self_tests() -> None:
    """
    Executes an automated self-test on an isolated temporary SQLite database file.

    Checks verified:
    1. Register a user, then login succeeds
    2. Login with a wrong password fails
    3. Registering a duplicate username fails
    4. Weak passwords and bad usernames are rejected
    5. Save a check, read it back, filter by risk level
    6. One user cannot delete another user's check
    7. get_stats returns sensible numbers
    8. The stored password hash is not the plain password
    """
    import os
    # Import AuthManager for integrated authentication tests
    from auth import AuthManager

    temp_db_path: Path = Path(__file__).resolve().parent / "data" / "temp_test_db.sqlite"
    if temp_db_path.is_file():
        temp_db_path.unlink()

    print("=" * 70)
    print("RUNNING DATABASE & AUTHENTICATION SELF-TEST SUITE")
    print(f"Temporary Test Database: {temp_db_path.name}")
    print("=" * 70)

    test_db = Database(db_path=temp_db_path)
    auth = AuthManager(db=test_db)

    def print_result(test_num: int, description: str, passed: bool) -> None:
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"Test {test_num}: {description.ljust(52)} {status_str}")
        if not passed:
            raise AssertionError(f"Test {test_num} failed: {description}")

    try:
        # Test 1: Register a user, then login succeeds
        user_alice = auth.register("alice_test", "SecurePass123")
        login_alice = auth.login("alice_test", "SecurePass123")
        t1_passed = (
            user_alice is not None
            and login_alice is not None
            and login_alice["username"] == "alice_test"
            and "password_hash" not in login_alice
        )
        print_result(1, "Register user, then login succeeds", t1_passed)

        # Test 2: Login with a wrong password fails
        login_wrong = auth.login("alice_test", "WrongPassword999")
        login_nonexistent = auth.login("nobody_exists", "SecurePass123")
        t2_passed = login_wrong is None and login_nonexistent is None
        print_result(2, "Login with wrong password fails generically", t2_passed)

        # Test 3: Registering a duplicate username fails
        t3_passed = False
        try:
            auth.register("alice_test", "AnotherPass456")
        except ValueError:
            t3_passed = True
        print_result(3, "Registering duplicate username raises ValueError", t3_passed)

        # Test 4: Weak passwords and bad usernames are rejected
        bad_inputs = [
            ("al", "SecurePass123"),            # Too short (< 3 chars)
            ("alice@home", "SecurePass123"),    # Invalid symbol (@)
            ("alice_test_2", "short1"),         # Password < 8 chars
            ("alice_test_2", "alllettersonly"), # Password lacks digits
            ("alice_test_2", "1234567890"),     # Password lacks letters
        ]
        t4_passed = True
        for u, p in bad_inputs:
            try:
                auth.register(u, p)
                t4_passed = False
            except ValueError:
                pass  # Correctly rejected
        print_result(4, "Weak passwords & invalid usernames rejected", t4_passed)

        # Test 5: Save a check, read it back, filter by risk level
        user_bob = auth.register("bob_test", "SecretCode456")
        alice_id = int(login_alice["id"])
        bob_id = int(user_bob["id"])

        c1_id = test_db.save_check(
            user_id=alice_id,
            job_title="Junior Python Developer",
            job_text="Build web scrapers and clean text data.",
            fraud_probability=0.08,
            risk_level="low",
            model_name="Logistic Regression",
            warnings=[],
            rule_flags={"missing_salary": True},
        )
        c2_id = test_db.save_check(
            user_id=alice_id,
            job_title="Urgent Data Entry Assistant",
            job_text="Earn $1000 daily with immediate registration fee.",
            fraud_probability=0.92,
            risk_level="high",
            model_name="Logistic Regression",
            warnings=[{"id": "fee_payment", "severity": "high"}],
            rule_flags={"missing_salary": True},
        )

        alice_checks = test_db.get_checks_for_user(alice_id)
        high_checks = test_db.get_checks_for_user(alice_id, risk_level="high")
        search_checks = test_db.get_checks_for_user(alice_id, search="Python")

        t5_passed = (
            len(alice_checks) == 2
            and len(high_checks) == 1
            and high_checks[0]["id"] == c2_id
            and len(search_checks) == 1
            and search_checks[0]["id"] == c1_id
            and isinstance(alice_checks[0]["warnings"], list)
        )
        print_result(5, "Save check, read back, and filter by risk level", t5_passed)

        # Test 6: One user cannot delete another user's check
        bob_del_attempt = test_db.delete_check(check_id=c1_id, user_id=bob_id)
        check_still_exists = test_db.get_check(c1_id) is not None
        alice_del_success = test_db.delete_check(check_id=c1_id, user_id=alice_id)
        check_now_deleted = test_db.get_check(c1_id) is None

        t6_passed = (
            bob_del_attempt is False
            and check_still_exists
            and alice_del_success is True
            and check_now_deleted
        )
        print_result(6, "User cannot delete another user's check (IDOR check)", t6_passed)

        # Test 7: get_stats returns sensible numbers
        stats = test_db.get_stats()
        t7_passed = (
            stats["total_users"] == 2
            and stats["total_checks"] == 1  # 1 remaining after c1 deleted
            and stats["high_risk_count"] == 1
            and stats["high_risk_percentage"] == 100.0
            and len(stats["checks_per_day"]) == 14
        )
        print_result(7, "get_stats returns consistent metrics & daily series", t7_passed)

        # Test 8: Stored password hash is not the plain password
        raw_user = test_db.get_user_by_username("bob_test")
        stored_hash = raw_user["password_hash"] if raw_user else ""
        t8_passed = (
            stored_hash != "SecretCode456"
            and stored_hash.startswith("$2b$")
            and len(stored_hash) >= 50
        )
        print_result(8, "Stored password is a secure bcrypt hash ($2b$)", t8_passed)

        # Test 9: get_all_checks returns checks with username joined
        all_checks = test_db.get_all_checks()
        t9_passed = (
            len(all_checks) == 1
            and all_checks[0]["id"] == c2_id
            and all_checks[0]["username"] == "alice_test"
            and "warnings" in all_checks[0]
            and isinstance(all_checks[0]["warnings"], list)
        )
        print_result(9, "get_all_checks joins username from users table", t9_passed)

        # Test 10: get_check_with_owner returns single check with username
        owner_check = test_db.get_check_with_owner(c2_id)
        none_check = test_db.get_check_with_owner(999999)
        t10_passed = (
            owner_check is not None
            and owner_check["id"] == c2_id
            and owner_check["username"] == "alice_test"
            and none_check is None
        )
        print_result(10, "get_check_with_owner retrieves check with owner username", t10_passed)

        # Test 11: Admin user management and deletion rules
        admin1 = auth.create_admin("admin_lead", "AdminPass123")
        admin2 = auth.create_admin("admin_sub", "AdminPass456")
        # Deleting one of two admins succeeds
        del_admin2 = test_db.delete_user(admin2["id"])
        # Deleting the last admin account is blocked with ValueError
        last_admin_blocked = False
        try:
            test_db.delete_user(admin1["id"])
        except ValueError:
            last_admin_blocked = True
        # Admin can delete any check by check_id
        admin_del_check = test_db.delete_check(check_id=c2_id)
        check_now_absent = test_db.get_check(c2_id) is None

        t11_passed = (
            del_admin2 is True
            and last_admin_blocked is True
            and admin_del_check is True
            and check_now_absent
        )
        print_result(11, "Admin rules: last admin protected & admin check delete", t11_passed)

    finally:
        # Clean up temporary database file
        if temp_db_path.is_file():
            try:
                temp_db_path.unlink()
            except PermissionError:
                pass

    print("=" * 70)
    print("ALL 11 SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_self_tests()
