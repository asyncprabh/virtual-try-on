import os

# List of directories to create
directories = [
    'static',
    'static/css',
    'static/js',
    'static/images',
    'static/uploads'
]

# Create each directory
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
