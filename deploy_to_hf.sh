#!/usr/bin/env bash
# Deploy to Hugging Face Spaces.
#
#   1. Sign up at https://huggingface.co/join  (use the username Racecarman0924 so the
#      README's live-demo link is already correct)
#   2. Create a Space: https://huggingface.co/new-space
#        Name : earnings-press-report-predictive-transformer
#        SDK  : Streamlit
#        Hardware: CPU basic (free)
#   3. Create a WRITE token: https://huggingface.co/settings/tokens
#   4. Run this script. Git will prompt for your username and the token as the password.
#
# Usage:  ./deploy_to_hf.sh <your-hf-username>

set -euo pipefail
USER="${1:?usage: ./deploy_to_hf.sh <your-hf-username>}"
SPACE="earnings-press-report-predictive-transformer"
WORK="$(mktemp -d)"

echo "==> cloning the Space"
git clone "https://huggingface.co/spaces/${USER}/${SPACE}" "$WORK/space"

echo "==> copying the application"
cd "$(dirname "$0")"
for p in app.py requirements.txt model_deploy.pt notebook.ipynb LICENSE model data out docs; do
  cp -R "$p" "$WORK/space/"
done

echo "==> writing the Space README (YAML front matter is required by HF)"
{
  printf -- '---\ntitle: Earnings Press Report Predictive Transformer\nemoji: "\U0001F4CA"\ncolorFrom: blue\ncolorTo: gray\nsdk: streamlit\napp_file: app.py\npinned: false\nlicense: mit\n---\n\n'
  cat README.md
} > "$WORK/space/README.md"

cd "$WORK/space"
git add -A
git commit -m "Deploy Earnings Press Report Predictive Transformer"
git push

echo
echo "==> live at https://huggingface.co/spaces/${USER}/${SPACE}"
