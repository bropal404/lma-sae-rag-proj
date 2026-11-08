#!/bin/sh
# git asks for credentials through this; the token stays in the environment, not in .git/config.
case "$1" in Username*) echo x-access-token ;; *) echo "$GH_TOKEN" ;; esac
