#!/bin/bash
echo "Update semantic-aqo..."

docker-compose -f ../docker-compose-dev.yml exec postgres bash -c "cd /usr/src/semantic-aqo && make clean && make install"
# docker-compose -f ../docker-compose-dev.yml restart postgres

echo "Done updating semantic-aqo..."
