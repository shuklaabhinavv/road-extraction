# Deploying the demo to Hugging Face Spaces (free)

One-time prerequisites (human steps):
1. Create a free account at https://huggingface.co (personal identity).
2. Settings -> Access Tokens -> New token (type: Write). Keep it handy.

Then the deploy (Claude or manual):

```bash
pip install -U huggingface_hub
huggingface-cli login            # paste the token

# create the Space (Docker SDK)
huggingface-cli repo create roadx --repo-type space --space_sdk docker

# assemble the Space contents
git clone https://huggingface.co/spaces/<username>/roadx hf-space
cd hf-space
cp -r ../src ../web ../runs ../pyproject.toml ../requirements.txt .
cp ../deploy/Dockerfile .
cp ../deploy/README-space.md README.md

# checkpoints are ~400 MB -> track with LFS
huggingface-cli lfs-enable-largefiles .
git lfs track "*.pt"
git add -A && git commit -m "roadx demo" && git push
```

The Space builds automatically; the app appears at
`https://huggingface.co/spaces/<username>/roadx` after ~10 minutes.

Notes:
- `ROADX_MAX_TILES=36` caps anywhere-mode regions at ~1536 px per side so
  CPU inference stays under ~60 s.
- Free Spaces sleep after inactivity; first visit after sleep takes ~1 min.
