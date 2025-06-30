#!/bin/bash
PORT=5000

# Check if the port is in use
if lsof -i :$PORT; then
    echo "Port $PORT is already in use. Terminating the process..."
    PID=$(lsof -t -i :$PORT)
    kill -9 $PID
    echo "Process $PID terminated."
else
    echo "Port $PORT is free."
fi

# Start the Flask app
flask run
