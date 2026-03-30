"""
MARS - Multi-Agent Research System
Entry point for direct execution.

Usage:
    python main.py search "federated learning privacy"
    python main.py full "federated learning with differential privacy"
    python main.py api       # Start FastAPI server
"""

from mars.cli import main

if __name__ == "__main__":
    main()
