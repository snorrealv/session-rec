#!/bin/sh 
FILE=$1

source .env

# Assert that the environment variables are set
test -z "$S3_ACCESS_KEY" && echo "S3_ACCESS_KEY is not set"
test -z "$S3_SECRET_KEY" && echo "S3_SECRET_KEY is not set"
test -z "$S3_REGION" && echo "S3_REGION is not set"
test -z "$S3_BUCKET" && echo "S3_BUCKET is not set"


curl --progress-bar -X PUT \
    --user "${S3_ACCESS_KEY}":"${S3_SECRET_KEY}" \
    --aws-sigv4 "aws:amz:${S3_REGION}:s3" \
    --upload-file ${FILE} \
    https://${S3_BUCKET}.s3.${S3_REGION}.amazonaws.com

