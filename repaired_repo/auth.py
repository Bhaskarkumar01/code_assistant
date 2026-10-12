import hashlib
import hmac

# Example stored user data with hashed password (hash of "1234")
stored_users = {
    "admin": "81dc9bdb52d04dc20036dbd8313ed055"  # MD5 hash of "1234"
}

def authenticate_user(username, password):
    if not isinstance(username, str) or not isinstance(password, str):
        return False

    stored_password_hash = stored_users.get(username)
    if not stored_password_hash:
        # User not found
        return False

    # Hash the provided password
    password_hash = hashlib.md5(password.encode()).hexdigest()

    # Use hmac.compare_digest to prevent timing attacks
    if hmac.compare_digest(password_hash, stored_password_hash):
        return True

    return False
