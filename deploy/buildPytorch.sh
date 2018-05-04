version="0.4.0.2"

service_name="gcr.io/ornate-axiom-187403/pytorch:$version"

echo "docker build -f Dockerfile --no-cache=true -t $service_name  ." 
docker build --no-cache=true -t $service_name  .

echo "gcloud docker -- push $service_name" 
gcloud docker -- push $service_name

