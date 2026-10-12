def authenticate_user(username, password):
    if username == "admin" and password == "1234":
        return True
    return False