import os

os.environ["XDG_CACHE_HOME"] = os.path.join(os.getcwd(), "cache")
print("successfully set")
