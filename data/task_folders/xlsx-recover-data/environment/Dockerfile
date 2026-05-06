FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Step 1: minimal apt install (just python + curl + tar utilities).
# Loop through CN mirrors so a single 502 doesn't kill the build.
RUN set -eux; \
    for mirror in \
        mirrors.ustc.edu.cn \
        mirror.sjtu.edu.cn \
        mirrors.cloud.tencent.com \
        mirrors.tuna.tsinghua.edu.cn \
        mirrors.aliyun.com \
        ports.ubuntu.com ; do \
      sed -i "s|http://[^/]*/ubuntu-ports|http://${mirror}/ubuntu-ports|g" /etc/apt/sources.list.d/ubuntu.sources ; \
      echo "=== trying mirror: ${mirror} ===" ; \
      if apt-get update && \
         apt-get install -y --no-install-recommends python3 python3-pip python3-venv curl xz-utils ca-certificates ; then \
        break ; \
      fi ; \
    done ; \
    command -v python3 && command -v curl && command -v xz ; \
    rm -rf /var/lib/apt/lists/*

# Step 2: install Node.js from the official tarball (via Clash proxy, bypasses apt).
# arm64 tarball is ~30MB; Clash -> nodejs.org is reliable.
RUN set -eux; \
    NODE_VERSION=v20.18.0 ; \
    ARCH=$(uname -m | sed 's/aarch64/arm64/; s/x86_64/x64/') ; \
    curl -fsSL --retry 3 "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-${ARCH}.tar.xz" -o /tmp/node.tar.xz ; \
    ls -la /tmp/node.tar.xz ; \
    mkdir -p /opt/node ; \
    tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1 ; \
    rm /tmp/node.tar.xz ; \
    ls -la /opt/node/bin/ ; \
    ln -sf /opt/node/bin/node /usr/local/bin/node ; \
    ln -sf /opt/node/bin/npm /usr/local/bin/npm ; \
    ln -sf /opt/node/bin/npx /usr/local/bin/npx ; \
    /usr/local/bin/node --version && /usr/local/bin/npm --version

# Step 3: pre-install claude-code via npmmirror (avoids downloads.claude.ai).
RUN npm config set registry https://registry.npmmirror.com/ \
    && npm install -g @anthropic-ai/claude-code \
    && ln -sf "$(npm root -g)/@anthropic-ai/claude-code/cli.js" /usr/local/bin/claude \
    && claude --version

# Create working directory
WORKDIR /root

# Copy the input Excel file to working directory
COPY nasa_budget_incomplete.xlsx /root/nasa_budget_incomplete.xlsx

# Copy skills to agent-specific locations
COPY skills /root/.claude/skills
COPY skills /root/.codex/skills
COPY skills /root/.opencode/skill
COPY skills /root/.goose/skills
COPY skills /root/.factory/skills
COPY skills /root/.agents/skills
COPY skills /root/.gemini/skills

# Install Python dependencies + uv (verifier test.sh uses uvx, github.com download often blocked)
RUN pip3 install --break-system-packages openpyxl==3.1.5 uv==0.9.7 pytest==8.4.1 pytest-json-ctrf==0.3.5 \
    && mkdir -p /root/.local/bin \
    && touch /root/.local/bin/env
