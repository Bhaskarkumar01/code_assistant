import time
import logging

logging.basicConfig(level=logging.WARNING)

def authenticate_user(username, password):
    # Simulate a user data store
    valid_users = {
        "admin": "1234"
    }
    
    # Introduce a fixed delay to mitigate brute force attacks
    time.sleep(0.5)
    
    # Check if username exists and password matches
    if username in valid_users and valid_users[username] == password:
        return True
    
    # Log the failed attempt without revealing which part was incorrect
    logging.warning(f"Failed login attempt for username: {username}")
    
    # Return False without revealing details
    return False
