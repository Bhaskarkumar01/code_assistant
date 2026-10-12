import pytest
from auth import authenticate_user

def test_authenticate_user_success():
    assert authenticate_user("admin", "1234") is True

def test_authenticate_user_wrong_password():
    assert authenticate_user("admin", "wrong") is False

def test_authenticate_user_wrong_username():
    assert authenticate_user("user", "1234") is False
