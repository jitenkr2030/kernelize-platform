#!/bin/bash
# GitHub Credential Helper Script
# This script securely handles GitHub authentication

if [ "$1" == "Username" ]; then
    echo "git"
elif [ "$1" == "Password" ]; then
    # Read token from environment variable or stdin
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "$GITHUB_TOKEN"
    else
        cat "$HOME/.github_token" 2>/dev/null
    fi
fi
