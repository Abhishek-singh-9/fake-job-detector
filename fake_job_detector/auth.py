"""
Authentication & Access Control Module for Fake Job Posting Detector.

Manages user identity, registration validation, credential verification, and role-based
authorization using standard bcrypt password hashing.

Security Best Practices for Viva:
- Password Hashing with Salt:
  Passwords are never stored in plaintext. `bcrypt.gensalt()` generates a random 128-bit
  salt, and the key derivation function applies 12 work rounds ($2b$12$) to resist
  brute-force and rainbow table attacks.
- Anti-User-Enumeration:
  The `login()` method returns an identical generic failure response (None) whether
  a username does not exist or the password was incorrect. Furthermore, a dummy bcrypt
  verification is run when a user is absent to prevent timing-based side-channel attacks.
- Password Complexity Enforcement:
  Enforces a minimum length of 8 characters containing both letters and numbers.
- No Hardcoded Credentials:
  The administrative CLI prompt (`python auth.py create-admin`) uses `getpass` to securely
  accept passwords from terminal input without echoing them to the screen.
"""

import getpass
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional
import bcrypt

# Ensure local sibling modules can be imported
SCRIPT_DIR: Path = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from database import Database

# Pre-computed dummy bcrypt hash used to equalize execution time on unknown usernames
DUMMY_BCRYPT_HASH: str = (
    "$2b$12$e8kPqK8o2Q/Z4V8yv9y5xeX1J8c1bL2d9eK5uP3sZ7aQ1fW9rT3v6"
)


class AuthManager:
    """
    Handles user registration, credential hashing, login verification, and roles.
    """

    def __init__(self, db: Optional[Database] = None) -> None:
        """
        Initializes the AuthManager with a Database instance.

        Args:
            db (Optional[Database]): Database instance. If None, instantiates default Database.
        """
        self.db: Database = db if db is not None else Database()

    @staticmethod
    def validate_username(username: str) -> None:
        """
        Validates username length and character set.

        Requirements:
        - 3 to 30 characters long
        - Letters (a-z, A-Z), digits (0-9), and underscores (_) only.

        Args:
            username (str): Candidate username.

        Raises:
            ValueError: If username fails validation rules.
        """
        if not username or not isinstance(username, str):
            raise ValueError("Username cannot be empty.")

        username_clean = username.strip()
        if len(username_clean) < 3 or len(username_clean) > 30:
            raise ValueError("Username must be between 3 and 30 characters in length.")

        if not re.match(r"^[a-zA-Z0-9_]+$", username_clean):
            raise ValueError(
                "Username may only contain alphanumeric characters and underscores (no spaces or symbols)."
            )

    @staticmethod
    def validate_password(password: str) -> None:
        """
        Validates password length and complexity.

        Requirements:
        - At least 8 characters
        - Contains at least one letter (a-z, A-Z)
        - Contains at least one digit (0-9)

        Args:
            password (str): Candidate password.

        Raises:
            ValueError: If password fails complexity rules.
        """
        if not password or not isinstance(password, str):
            raise ValueError("Password cannot be empty.")

        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long.")

        if not re.search(r"[a-zA-Z]", password):
            raise ValueError("Password must contain at least one letter.")

        if not re.search(r"[0-9]", password):
            raise ValueError("Password must contain at least one digit (number).")

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Generates a secure salted bcrypt hash for a plaintext password.

        Args:
            password (str): Plaintext password.

        Returns:
            str: Salted bcrypt hash string.
        """
        # Encode string to UTF-8 bytes for bcrypt
        pwd_bytes = password.encode("utf-8")
        # Generate random salt with default 12 work rounds
        salt = bcrypt.gensalt()
        hashed_bytes = bcrypt.hashpw(pwd_bytes, salt)
        return hashed_bytes.decode("utf-8")

    def register(self, username: str, password: str) -> Dict[str, Any]:
        """
        Validates, hashes, and registers a new standard user account.

        Args:
            username (str): Desired username.
            password (str): Plaintext password meeting complexity criteria.

        Returns:
            Dict[str, Any]: Sanitized user dictionary (without password_hash).

        Raises:
            ValueError: If username or password fail validation or username is taken.
        """
        username_clean = username.strip()

        # Validate inputs before attempting database operation
        self.validate_username(username_clean)
        self.validate_password(password)

        # Hash password with bcrypt
        password_hash = self.hash_password(password)

        # Insert user with standard 'user' role
        user_id = self.db.add_user(
            username=username_clean,
            password_hash=password_hash,
            role="user",
        )

        return {
            "id": user_id,
            "username": username_clean,
            "role": "user",
        }

    def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticates user credentials against the database.

        Security Note for Viva:
        Returns None for both non-existent usernames and incorrect passwords.
        This generic failure prevents attackers from harvesting registered usernames.
        A dummy bcrypt verification runs even when the username is missing to
        ensure constant-time response profiles.

        Args:
            username (str): Claimed username.
            password (str): Plaintext password to verify.

        Returns:
            Optional[Dict[str, Any]]: User profile dict (excluding password_hash) if valid, else None.
        """
        if not username or not password:
            return None

        username_clean = username.strip()
        user_record = self.db.get_user_by_username(username_clean)

        # If user does not exist, perform dummy hash verification to mitigate timing attacks
        if user_record is None:
            bcrypt.checkpw(password.encode("utf-8"), DUMMY_BCRYPT_HASH.encode("utf-8"))
            return None

        # Verify plaintext password against stored salted bcrypt hash
        stored_hash = user_record["password_hash"].encode("utf-8")
        candidate_bytes = password.encode("utf-8")

        if bcrypt.checkpw(candidate_bytes, stored_hash):
            # Return safe user profile without the password hash
            return {
                "id": user_record["id"],
                "username": user_record["username"],
                "role": user_record["role"],
                "created_at": user_record["created_at"],
            }

        return None

    @staticmethod
    def is_admin(user: Optional[Dict[str, Any]]) -> bool:
        """
        Checks whether the given user object has administrative privileges.

        Args:
            user (Optional[Dict[str, Any]]): User dictionary.

        Returns:
            bool: True if role is 'admin', False otherwise.
        """
        return bool(user and user.get("role") == "admin")

    def create_admin(self, username: str, password: str) -> Dict[str, Any]:
        """
        Creates an administrative account with role 'admin'.

        Args:
            username (str): Desired admin username.
            password (str): Plaintext password meeting complexity criteria.

        Returns:
            Dict[str, Any]: Sanitized user dictionary (without password_hash).

        Raises:
            ValueError: If credentials fail validation or username is taken.
        """
        username_clean = username.strip()
        self.validate_username(username_clean)
        self.validate_password(password)

        password_hash = self.hash_password(password)
        user_id = self.db.add_user(
            username=username_clean,
            password_hash=password_hash,
            role="admin",
        )

        return {
            "id": user_id,
            "username": username_clean,
            "role": "admin",
        }


# -----------------------------------------------------------------------------
# CLI Utility: Create Administrator
# -----------------------------------------------------------------------------

def handle_create_admin_cli() -> None:
    """
    Interactive command-line workflow to provision an administrator.
    Invoked with: python auth.py create-admin
    """
    print("=" * 60)
    print("CREATE ADMINISTRATOR ACCOUNT")
    print("=" * 60)

    try:
        username = input("Enter admin username: ").strip()
        password = getpass.getpass("Enter admin password (min 8 chars, 1 letter, 1 digit): ")
        password_confirm = getpass.getpass("Confirm admin password: ")

        if password != password_confirm:
            print("\nError: Passwords do not match. Administrator creation aborted.")
            sys.exit(1)

        auth_mgr = AuthManager()
        admin_user = auth_mgr.create_admin(username=username, password=password)

        print(f"\nSuccess: Administrator '{admin_user['username']}' created successfully (User ID: {admin_user['id']}).")
        print("=" * 60)

    except ValueError as e:
        print(f"\nValidation Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    # Check if 'create-admin' argument was supplied
    if len(sys.argv) > 1 and sys.argv[1] == "create-admin":
        handle_create_admin_cli()
    else:
        print("Usage:")
        print("  python auth.py create-admin   -> Interactively creates an administrator account.")
        print("  python database.py            -> Runs the automated test suite for database and auth.")
