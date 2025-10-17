#!/bin/sh
# init-qdrant.sh - Pre-create Qdrant collections with correct dimensions
# CLAUDE FIX: Only creates openmemory collection - mem0migrations auto-created by mem0

echo "=========================================="
echo "Qdrant Collection Initialization"
echo "=========================================="

echo "Waiting for Qdrant to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

# Wait for Qdrant
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Attempt $RETRY_COUNT/$MAX_RETRIES..."
    
    if curl -f -s --max-time 2 http://localhost:6333/collections >/dev/null 2>&1; then
        echo "✓ Qdrant is ready!"
        break
    fi
    
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "✗ Qdrant failed to start!"
    curl -v http://localhost:6333/collections 2>&1 | head -20
    exit 1
fi

# CLAUDE FIX: Only create openmemory - let mem0 create mem0migrations with its own dimensions (1536)
echo ""
echo "----------------------------------------"
echo "Creating openmemory collection"
echo "----------------------------------------"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/collections/openmemory)

if [ "$STATUS" = "200" ]; then
    echo "✓ Collection 'openmemory' already exists"
else
    echo "Creating 'openmemory' with 1024 dimensions..."
    
    RESPONSE=$(curl -s -X PUT \
        -H "Content-Type: application/json" \
        -d '{"vectors":{"size":1024,"distance":"Cosine"}}' \
        -w "\nHTTP:%{http_code}" \
        http://localhost:6333/collections/openmemory)
    
    if echo "$RESPONSE" | grep -q "HTTP:200"; then
        echo "✓ Created 'openmemory' with 1024 dimensions"
    else
        echo "✗ Failed to create 'openmemory'"
        echo "Response: $RESPONSE"
    fi
fi

echo ""
echo "=========================================="
echo "Initialization Complete!"
echo "=========================================="
echo "Collections created:"
curl -s http://localhost:6333/collections | grep -o '"name":"[^"]*"' || echo "Failed to list"
echo ""
echo "Note: mem0migrations will be auto-created by mem0"
echo "      with its required dimensions (1536)"
echo "=========================================="