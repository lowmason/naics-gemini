#!/bin/bash
# Setup script specifically for Lambda Labs instances
# Lambda Labs instances typically come with Ubuntu and basic tools pre-installed
# This script sets up the NAICS Embedder project environment

set -e

echo "Setting up NAICS Embedder project on Lambda Labs..."

# Lambda Labs instances usually have git pre-installed, but check anyway
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    sudo apt-get update && sudo apt-get install -y git
fi

# Install uv if not installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
    if [ -f "$HOME/.zshrc" ]; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.zshrc"
    fi
    echo "uv installed"
else
    echo "uv is already installed: $(uv --version)"
fi

# Ensure uv is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Configure git user (if not already set)
# Lambda Labs instances typically don't have git configured
if [ -z "$(git config --global user.name)" ]; then
    if [ -n "$GIT_AUTHOR_NAME" ]; then
        git_name="$GIT_AUTHOR_NAME"
        echo "Using GIT_AUTHOR_NAME from environment: $git_name"
    elif [ -n "$GIT_COMMITTER_NAME" ]; then
        git_name="$GIT_COMMITTER_NAME"
        echo "Using GIT_COMMITTER_NAME from environment: $git_name"
    else
        echo "Git user.name is not set."
        read -p "Enter your git user.name: " git_name
    fi
    if [ -n "$git_name" ]; then
        git config --global user.name "$git_name"
        echo "Set git user.name to: $git_name"
    fi
else
    echo "Git user.name is already set: $(git config --global user.name)"
fi

if [ -z "$(git config --global user.email)" ]; then
    if [ -n "$GIT_AUTHOR_EMAIL" ]; then
        git_email="$GIT_AUTHOR_EMAIL"
        echo "Using GIT_AUTHOR_EMAIL from environment: $git_email"
    elif [ -n "$GIT_COMMITTER_EMAIL" ]; then
        git_email="$GIT_COMMITTER_EMAIL"
        echo "Using GIT_COMMITTER_EMAIL from environment: $git_email"
    else
        echo "Git user.email is not set."
        read -p "Enter your git user.email: " git_email
    fi
    if [ -n "$git_email" ]; then
        git config --global user.email "$git_email"
        echo "Set git user.email to: $git_email"
    fi
else
    echo "Git user.email is already set: $(git config --global user.email)"
fi

# Install zsh if not installed
if ! command -v zsh &> /dev/null; then
    echo "Installing zsh..."
    sudo apt-get update && sudo apt-get install -y zsh
    echo "zsh installed"
else
    echo "zsh is already installed: $(zsh --version)"
fi

# Install oh-my-zsh if not installed
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "Installing oh-my-zsh..."
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
    echo "oh-my-zsh installed"
else
    echo "oh-my-zsh is already installed"
fi

# Install Spaceship theme if not installed
if [ ! -d "$HOME/.oh-my-zsh/custom/themes/spaceship-prompt" ]; then
    echo "Installing Spaceship theme..."
    git clone https://github.com/spaceship-prompt/spaceship-prompt.git "$HOME/.oh-my-zsh/custom/themes/spaceship-prompt" --depth=1
    ln -s "$HOME/.oh-my-zsh/custom/themes/spaceship-prompt/spaceship.zsh-theme" "$HOME/.oh-my-zsh/custom/themes/spaceship.zsh-theme"
    echo "Spaceship theme installed"
else
    echo "Spaceship theme is already installed"
fi

# Create or update .zshrc with Spaceship configuration
if [ ! -f "$HOME/.zshrc" ] || ! grep -q "spaceship" "$HOME/.zshrc"; then
    echo "Configuring .zshrc with Spaceship theme..."
    
    # Backup existing .zshrc if it exists
    if [ -f "$HOME/.zshrc" ]; then
        cp "$HOME/.zshrc" "$HOME/.zshrc.backup.$(date +%Y%m%d_%H%M%S)"
        echo "Backed up existing .zshrc"
    fi
    
    # Create .zshrc with Spaceship configuration
    cat > "$HOME/.zshrc" << 'EOF'
# Add uv to PATH (if installed via official installer)
export PATH="$HOME/.cargo/bin:$PATH"

# Path to oh-my-zsh installation
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load
ZSH_THEME="spaceship"

# Plugins (add your preferred plugins here)
plugins=(
  git
  docker
  docker-compose
  python
  pip
  virtualenv
)

# Load oh-my-zsh
source $ZSH/oh-my-zsh.sh

# Spaceship theme configuration
SPACESHIP_PROMPT_ORDER=(
  user          # Username section
  dir           # Current directory section
  host          # Hostname section
  git           # Git section (git_branch + git_status)
  exec_time     # Execution time
  line_sep      # Line break
  vi_mode       # Vi-mode indicator
  jobs          # Background jobs indicator
  exit_code     # Exit code section
  char          # Prompt character
)

SPACESHIP_USER_SHOW=always
SPACESHIP_PROMPT_ADD_NEWLINE=false
SPACESHIP_CHAR_SYMBOL="❯ "
SPACESHIP_CHAR_SUFFIX=" "

# Git configuration (if not set globally)
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-$(git config --global user.name)}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-$(git config --global user.email)}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"

# Load local customizations if they exist
[ -f "$HOME/.zshrc.local" ] && source "$HOME/.zshrc.local"
EOF
    echo "Created .zshrc with Spaceship configuration"
else
    echo ".zshrc already configured"
fi

# Set zsh as default shell (if not already)
CURRENT_SHELL=$(basename "$SHELL")
if [ "$CURRENT_SHELL" != "zsh" ]; then
    echo "Setting zsh as default shell..."
    ZSH_PATH=$(which zsh)
    if [ -n "$ZSH_PATH" ]; then
        chsh -s "$ZSH_PATH" || echo "Could not change default shell. You may need to run: chsh -s $(which zsh)"
        echo "zsh will be your default shell on next login"
    fi
else
    echo "zsh is already your default shell"
fi

# Set up project-specific git config
if [ -f ".git/config" ]; then
    git config --global --add safe.directory "$(pwd)" 2>/dev/null || true
    echo "Configured git safe directory"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Current configuration:"
echo "  Shell: $(basename $SHELL) (will be zsh on next login)"
echo "  Git user.name:  $(git config --global user.name)"
echo "  Git user.email: $(git config --global user.email)"
echo ""
echo "To activate zsh now, run: zsh"
echo "Or log out and log back in to use zsh as your default shell."
echo ""

