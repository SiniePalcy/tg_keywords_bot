#!/bin/bash

LINES=${1:-30}

ssh tgbot "journalctl -u tgkeywordsbot -n $LINES -f"
