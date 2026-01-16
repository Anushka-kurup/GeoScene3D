#!/bin/bash
# Work with Partial Replica Dataset Downloads
# Use parts a, b, c even while download continues

echo "🔧 Working with Partial Replica Files"
echo "======================================"

# Check which parts exist
echo ""
echo "📦 Checking available parts..."
PARTS_DIR="."
PARTS_FOUND=0

for part in partaa partab partac partad partae partaf partag partah partai; do
    if [ -f "replica_v1_0.tar.gz.$part" ]; then
        echo "  ✅ Found: replica_v1_0.tar.gz.$part"
        PARTS_FOUND=$((PARTS_FOUND + 1))
    else
        echo "  ⏳ Missing: replica_v1_0.tar.gz.$part (still downloading...)"
    fi
done

echo ""
echo "Total parts available: $PARTS_FOUND"

# If we have at least 3 parts (a, b, c), we can extract some scenes
if [ $PARTS_FOUND -ge 3 ]; then
    echo ""
    echo "🎉 You have enough parts to start!"
    echo ""
    echo "Option 1: Extract what we can now"
    echo "---------------------------------"
    echo "cat replica_v1_0.tar.gz.parta* | tar xz 2>/dev/null"
    echo ""
    echo "This will extract whatever is complete in the first 3 parts."
    echo ""
    echo "Option 2: Wait for all parts, then extract everything"
    echo "----------------------------------------------------"
    echo "Once all parts download, the script will automatically:"
    echo "  cat replica_v1_0.tar.gz.part* | tar xz"
    echo ""
    
    read -p "Do you want to try extracting now? (y/n): " choice
    
    if [ "$choice" = "y" ]; then
        echo ""
        echo "🔄 Attempting partial extraction..."
        cat replica_v1_0.tar.gz.parta* 2>/dev/null | tar xz 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "✅ Partial extraction successful!"
            echo ""
            echo "📁 Checking what we got:"
            ls -lh replica_v1/ 2>/dev/null || ls -lh 
        else
            echo "⚠️  Partial extraction incomplete - some files may be corrupted"
            echo "💡 Recommendation: Wait for all parts to finish downloading"
        fi
    fi
else
    echo ""
    echo "⏳ Need at least 3 parts to try extraction."
    echo "   Current: $PARTS_FOUND parts"
    echo "   Please wait for more parts to download..."
fi

echo ""
echo "📊 To check download progress:"
echo "   ls -lh replica_v1_0.tar.gz.part*"
