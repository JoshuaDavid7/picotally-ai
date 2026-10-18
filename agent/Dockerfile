# Production image for the PicoTally voice agent worker.
#
# Runs the LiveKit Agents worker (Gemini 3.1 Flash Live) that joins voice rooms
# and bridges tool calls back to the voice_agent_tools edge function. Works both
# for LiveKit Cloud Agents (the recommended host — co-located with the media
# servers for low latency) and any container host (Fly.io / Render / a VM).
#
# Secrets (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, GOOGLE_API_KEY,
# SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY) are injected by the
# host's secret manager at runtime — never baked into the image. See DEPLOY.md.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Prefetch any model files the plugins need so cold starts are fast. Harmless
# no-op for the Gemini native-audio path (it does turn detection server-side).
RUN python main.py download-files || true

# `start` is the production worker mode (registers + waits for room dispatch);
# `dev` is local-only (file-watching hot reload).
CMD ["python", "main.py", "start"]
