# Push vector-db-stack to GitHub

## 1. Create a new repository on GitHub
- Go to https://github.com/new
- Name: vector-db-stack
- Description: "Vector database stack for development and testing"
- Make it public or private as needed
- DON'T initialize with README (we already have one)

## 2. Add remote and push
```bash
# Add the remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/vector-db-stack.git

# Push main branch
git checkout main
git push -u origin main

# Push development branch
git checkout development
git push -u origin development
```

## Current Status
- main branch: Initial setup
- development branch: Updated configuration (current)
- No Claude references in commits