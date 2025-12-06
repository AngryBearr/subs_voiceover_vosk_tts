# Set strict error handling: stop the script on the first error.
$ErrorActionPreference = "Stop"

# --- Script to configure a local Git ignore file ---
# Sets a repository-local excludes file using Git's core.excludesfile

# Step 1: Determine repository root
try {
    # Use git to find the repository root
    $REPO_ROOT = git rev-parse --show-toplevel
}
catch {
    # If the command fails, we are likely not inside a Git repository
    Write-Host "❌ Could not determine Git repository. Are you inside one?" -ForegroundColor Red
    # Exit with an error code
    exit 1
}

# Build the path to the local excludes file
$LOCAL_IGNORE_FILE = Join-Path -Path $REPO_ROOT -ChildPath ".local_gitignore"

Write-Host "📍 Repository root: $REPO_ROOT"

# Step 2: Configure local Git setting
# Set core.excludesfile so Git uses the local ignore file
git config --local core.excludesfile $LOCAL_IGNORE_FILE
Write-Host "✅ Local git config core.excludesfile set → $LOCAL_IGNORE_FILE"

# Step 3: Create .local_gitignore with example rules if missing
if (-not (Test-Path -Path $LOCAL_IGNORE_FILE -PathType Leaf)) {
    # Create a multi-line example content using a here-string
    $fileContent = @"
# Local gitignore - rules here apply only to your local environment
# and will not be included in commits.

# Ignore this file itself
.local_gitignore

# Example rules:
*.log
node_modules/
.env
"@
    # Write the example content to the file
    Set-Content -Path $LOCAL_IGNORE_FILE -Value $fileContent
    Write-Host "✅ Created file $LOCAL_IGNORE_FILE with example rules."
}
else {
    Write-Host "ℹ️ File $LOCAL_IGNORE_FILE already exists."
}

# Step 4: Final confirmation
Write-Host ""
Write-Host "🎯 Done! Your local ignore file is ready." -ForegroundColor Green
Write-Host "   You can edit it at: $LOCAL_IGNORE_FILE"
Write-Host ""
Write-Host "💡 Changes in this file affect only your local repository and will not modify the shared .gitignore tracked by commits."
