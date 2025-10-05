#!/bin/sh

# Generate a runtime config.js file from the environment variable
echo "window.env = {" > /usr/share/nginx/html/config.js
echo "  API_BASE_URL: '${API_BASE_URL:-http://localhost:5000/api}'," >> /usr/share/nginx/html/config.js
echo "};" >> /usr/share/nginx/html/config.js

# Start nginx
exec nginx -g 'daemon off;'
