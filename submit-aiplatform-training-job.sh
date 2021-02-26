#!/bin/sh

set -ex

SEED=${SEED:-1}
JOB_ID="shibata_wscmaes_bench_seed${SEED}_$(date '+%Y%m%d%H%M%S')"

if [ -z "$IMAGE_URI" ] ; then
  echo "No webhook url"
  exit 1
fi

if [ -z "$SLACK_WEBHOOK_URL" ] ; then
  echo "No webhook url"
  exit 1
fi

if [ -z "$SLACK_WEBHOOK_CHANNEL" ] ; then
  echo "No webhook channel"
  exit 1
fi

if [ -z "$GCS_PATH" ] ; then
  echo "No gcs path"
  exit 1
fi

docker build -t $IMAGE_URI .
docker push $IMAGE_URI

gcloud ai-platform jobs submit training $JOB_ID \
  --region asia-northeast1 \
  --master-image-uri $IMAGE_URI \
  --scale-tier CUSTOM \
  --master-machine-type n1-standard-16 \
  -- \
  python ./benchmark_aiplatform.py \
  --seed $SEED \
  --job-id $JOB_ID \
  --slack-url $SLACK_WEBHOOK_URL \
  --slack-channel $SLACK_WEBHOOK_CHANNEL \
  --gcs-path $GCS_PATH