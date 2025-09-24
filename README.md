```bash
# Local Dev
uvicorn main:app --reload --port 5005

# Docker
docker build -t reddit-scrapper:latest .

# Docker Dev
docker run -p 5005:5005 reddit-scrapper:latest

# Push to Harbour
docker build -t reddit-scrapper:latest .
docker tag reddit-scrapper:latest ihl-harbor.apps.innovate.sg-cna.com/smu/reddit-scrapper:latest
docker push ihl-harbor.apps.innovate.sg-cna.com/smu/reddit-scrapper:latest
```