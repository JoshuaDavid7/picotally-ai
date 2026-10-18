# Running the voice worker

The worker is a LiveKit Agents process. It joins a LiveKit room when the app
requests one, runs Gemini Live for speech in and speech out, and calls the
`voice_agent_tools` edge function for every tool so that voice and text share
one deterministic tool layer.

Required environment (never committed):

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `GOOGLE_API_KEY` for Gemini Live
- `SUPABASE_URL` and `VOICE_AGENT_SECRET`, the dedicated shared secret the
  tools endpoint checks in constant time

Local run: `pip install -r requirements.txt` then `python main.py dev`.
Production runs one always-warm instance in a container (see `Dockerfile`);
the hosting details are not part of this repository.
