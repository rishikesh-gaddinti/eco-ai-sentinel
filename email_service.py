import json
import os

DB_FILE = "subscribers.json"

def get_subscribers():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        return json.load(f)

def add_subscriber(email):
    subs = get_subscribers()
    if email not in subs and "@" in email:
        subs.append(email)
        with open(DB_FILE, "w") as f:
            json.dump(subs, f)
        return True
    return False

def remove_subscriber(email):
    subs = get_subscribers()
    if email in subs:
        subs.remove(email)
        with open(DB_FILE, "w") as f:
            json.dump(subs, f)
        return True
    return False
