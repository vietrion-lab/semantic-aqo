#!/bin/bash
#
# This script connects to the psql shell inside the 'postgres' service container,
# explicitly using the correct user.
#

echo "Connecting to psql in container (service: postgres)..."

# Load environment variables from .env file
# This makes the script aware of your POSTGRES_USER
set -a
[ -f .env ] && . .env
set +a

# Use the POSTGRES_USER from .env, or default to 'postgres'
PG_USER=${POSTGRES_USER:-postgres}

echo "Logging in as user: $PG_USER"

# We add the "-U $PG_USER" flag to tell psql which user to connect as
docker compose -f ../docker-compose-dev.yml exec postgres psql -U $PG_USER
