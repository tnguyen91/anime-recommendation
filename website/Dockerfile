# Multi-stage build for better security and smaller image size
FROM nginx:1.25-alpine as production

# Set labels for better image management
LABEL maintainer="anime-recommendation-team"
LABEL version="1.0.0"
LABEL description="Anime Recommendation UI"

# Install security updates
RUN apk update && apk upgrade && apk add --no-cache \
    ca-certificates \
    && rm -rf /var/cache/apk/*

# Create nginx user with specific UID/GID for consistency
RUN addgroup -g 1001 -S nginx-app && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G nginx-app -g nginx-app nginx-app

# Remove default nginx website and create necessary directories
RUN rm -rf /usr/share/nginx/html/* && \
    rm -f /etc/nginx/conf.d/default.conf && \
    mkdir -p /var/cache/nginx && \
    chown -R nginx-app:nginx-app /var/cache/nginx /var/log/nginx

# Copy custom nginx configuration
COPY --chown=nginx-app:nginx-app nginx.conf /etc/nginx/nginx.conf

# Copy website files with proper ownership
COPY --chown=nginx-app:nginx-app *.html *.css *.js /usr/share/nginx/html/

# Set proper permissions
RUN chmod -R 644 /usr/share/nginx/html/* && \
    chmod 755 /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

# Switch to non-root user
USER nginx-app

# Expose port
EXPOSE 80

# Start nginx with proper signal handling
STOPSIGNAL SIGQUIT
CMD ["nginx", "-g", "daemon off;"]