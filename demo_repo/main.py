from auth import authenticate_user

def login(username, password):
    return authenticate_user(username, password)