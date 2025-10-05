# Use nginx to serve static frontend files
FROM nginx:alpine

# Copy frontend files into nginx html directory
COPY . /usr/share/nginx/html

# Copy startup script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Start nginx via entrypoint (this will generate config.js)
ENTRYPOINT ["/entrypoint.sh"]
