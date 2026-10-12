import pytest
from auth import authenticate_user

def test_authenticate_user_success():
    assert authenticate_user("admin", "1234") is True

def test_authenticate_user_wrong_password():
    assert authenticate_user("admin", "wrong") is False

def test_authenticate_user_unknown_user():
    assert authenticate_user("unknown", "1234") is False

def test_authenticate_user_invalid_input():
    assert authenticate_user(None, "1234") is False
    assert authenticate_user("admin", None) is False
    assert authenticate_user(123, 1234) is False
