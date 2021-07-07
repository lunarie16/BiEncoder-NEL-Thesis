#!/usr/bin/env bash

# build image
IMAGE=registry.datexis.com/mmenke/biencoder-nel

version=0.2.28
echo "Version: $version"
#docker login -u $1 -p $2 registry.datexis.com
docker build -t $IMAGE -t $IMAGE:$version ../.
docker push $IMAGE:$version
echo "Done pushing image $image for build $version"