import os

# check if ./cache exists if not create
if not os.path.exists(os.path.join(os.getcwd(), "cache")):
    os.makedirs(os.path.join(os.getcwd(), "cache"))

os.environ["XDG_CACHE_HOME"] = os.path.join(os.getcwd(), "cache")
print("successfully set")
