FROM python:3.12.5-slim

WORKDIR /app

COPY . /app/

# update and upgrade the packages and install build-base for sikit_learn
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
 build-essential \
 curl \
 software-properties-common \ 
 llvm \
&& rm -rf /var/lib/apt/lists/*

# Install the required packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 80

# Healthcheck for docker and portainer use to see if all is ok
HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health

# Run app.py when the container launches
CMD ["streamlit", "run", "TL.py", "--server.port=80", "--server.address=0.0.0.0", "--server.maxUploadSize=512", "--client.toolbarMode=minimal" ]
