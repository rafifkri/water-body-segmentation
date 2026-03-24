#!/usr/bin/env python
"""
Wrapper script - calls src.train for backward compatibility
For the actual implementation, see src/train.py
"""
import sys
from pathlib import Path

# Run the training module
if __name__ == '__main__':
    from src.train import main
    main()

