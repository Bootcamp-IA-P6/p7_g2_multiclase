FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# set directory
WORKDIR /app

# set config
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# download system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copy dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# copy files
COPY . .

# install project
RUN uv sync --frozen --no-dev

# expose port
EXPOSE 8501

# command: uv run streamlit run app.main.py
CMD ["uv", "run", "streamlit", "run", "app/main.py"]