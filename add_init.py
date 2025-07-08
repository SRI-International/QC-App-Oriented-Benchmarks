import os

EXCLUDED_DIRS = {"_containerbuildfiles", "_doc", "_setup", ".github", }

for root, dirs, files in os.walk("."):
    # Remove excluded dirs from the dirs list *in-place*
    dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

    # Add the init file if it does not exist along the walk
    init_file = os.path.join(root, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            pass  # creates an empty __init__.py
