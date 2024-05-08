'''
C

Author: Matej Vadovic
Year: 2024
'''
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add the current directory to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)