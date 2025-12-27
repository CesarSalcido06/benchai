#!/bin/bash
# Install BenchAI systemd services

set -e

echo "=== Installing BenchAI Services ==="

# Copy service files
sudo cp /home/user/benchai/systemd/benchai-llm.service /etc/systemd/system/
sudo cp /home/user/benchai/systemd/benchai-router.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable benchai-llm.service
sudo systemctl enable benchai-router.service

echo ""
echo "=== Services Installed ==="
echo ""
echo "To start services:"
echo "  sudo systemctl start benchai-llm"
echo "  sudo systemctl start benchai-router"
echo ""
echo "To check status:"
echo "  sudo systemctl status benchai-llm"
echo "  sudo systemctl status benchai-router"
echo ""
echo "To view logs:"
echo "  journalctl -u benchai-router -f"
