 # Use the official Jupyter base image (includes Python 3)
FROM jupyter/base-notebook:latest

# Install the requested data science packages
# We use --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn
