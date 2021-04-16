FROM python:3.8.5
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt 
COPY . .
EXPOSE 80
CMD ["python", "app.py"]